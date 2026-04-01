from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import threading
from pathlib import Path

import torch
from types import ModuleType
from typing import Callable, Dict, Iterable, Tuple

logger = logging.getLogger("baymax-zimage")

_PATCH_LOCK = threading.Lock()
_ORIGINAL_FUNCTIONS: Dict[Tuple[str, str], Callable] = {}
_PACKAGE_DIR = Path(__file__).resolve().parent
_USER_IMPL_PATH = _PACKAGE_DIR / "user_impl.py"


def _target_name(target: object) -> str:
    module_name = getattr(target, "__module__", target.__class__.__module__)
    qualname = getattr(target, "__qualname__", target.__class__.__qualname__)
    return f"{module_name}.{qualname}"


def _available_targets() -> Tuple[Tuple[object, str], ...]:
    targets = []
    for target, attribute_name in _target_modules():
        if hasattr(target, attribute_name):
            targets.append((target, attribute_name))
        else:
            logger.warning(
                "[baymax-zimage] Skipping %s.%s because it does not exist",
                _target_name(target),
                attribute_name,
            )
    return tuple(targets)


def _target_modules() -> Iterable[Tuple[object, str]]:
    ops = importlib.import_module("comfy.ops")
    targets = []

    disable_weight_init = getattr(ops, "disable_weight_init", None)
    if disable_weight_init is not None and hasattr(disable_weight_init, "RMSNorm"):
        targets.append((disable_weight_init.RMSNorm, "forward"))

    manual_cast = getattr(ops, "manual_cast", None)
    if manual_cast is not None and hasattr(manual_cast, "RMSNorm"):
        targets.append((manual_cast.RMSNorm, "forward"))

    fp8_ops = getattr(ops, "fp8_ops", None)
    if fp8_ops is not None and hasattr(fp8_ops, "RMSNorm"):
        targets.append((fp8_ops.RMSNorm, "forward"))

    return tuple(targets)


_USER_IMPL_MODULE_NAME = "baymax_zimage_user_impl"


def _load_user_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_USER_IMPL_MODULE_NAME, _USER_IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load user implementation from {_USER_IMPL_PATH}")

    module = importlib.util.module_from_spec(spec)
    # Register *before* exec so that any internal self-imports resolve correctly,
    # and so torch.compiler / Dynamo can locate the module by name during tracing.
    sys.modules[_USER_IMPL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _resolve_call_convention(user_function: Callable, original_function: Callable) -> Callable:
    """Inspect the user function signature *once at patch-install time* and return
    a thin caller that Dynamo can trace without graph breaks.

    The returned caller has the signature ``(module_self, x)`` and forwards to
    user_function using only attribute look-ups and tensor ops — nothing that
    requires dynamic Python evaluation on each call.
    """
    sig = inspect.signature(user_function)
    params = sig.parameters
    positional_slots = sum(
        1 for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    has_varkw   = any(p.kind == inspect.Parameter.VAR_KEYWORD   for p in params.values())

    want_original = "original_apply_rmsnorm" in params or has_varkw
    want_eps      = "eps"    in params or has_varkw
    want_module   = "module" in params or has_varkw

    if want_original or want_eps or want_module:
        # Keyword path — still passes the fallback/eps/module by name.
        # The fallback closure is a non-tensor constant; Dynamo treats it as a
        # compile-time constant and does NOT trace into it unless called, so
        # user_function itself (the tensor math) is still fully compiled.
        def caller(module_self, x):
            kw: dict = {}
            if want_original:
                def _fb(x_in, weight_in=None):
                    return original_function(module_self, x_in)
                kw["original_apply_rmsnorm"] = _fb
            if want_eps:
                kw["eps"] = module_self.eps
            if want_module:
                kw["module"] = module_self
            return user_function(x, module_self.weight, **kw)

    elif positional_slots >= 3 or has_varargs:
        # Three-positional path: (x, weight, fallback)
        def caller(module_self, x):
            def _fb(x_in, weight_in=None):
                return original_function(module_self, x_in)
            return user_function(x, module_self.weight, _fb)

    elif positional_slots == 2:
        # Pure two-argument path: (x, weight) — fully traceable, no fallback.
        def caller(module_self, x):
            return user_function(x, module_self.weight)

    else:
        # Single-argument path: (x,)
        def caller(module_self, x):
            return user_function(x)

    return caller


def _make_wrapper(module_name: str, user_function: Callable) -> Callable:
    original_function = _ORIGINAL_FUNCTIONS[(module_name, "forward")]

    # Resolve calling convention *once* at install time — the hot path
    # (patched_forward) then contains only a direct call forwarded through
    # ``caller``, so Dynamo can trace straight through into user_function
    # without any graph breaks from reflect/inspect calls.
    caller = _resolve_call_convention(user_function, original_function)

    def patched_forward(self, x):
        return caller(self, x)

    patched_forward.__name__ = "forward"
    patched_forward.__module__ = __name__
    return patched_forward


def _patch_rmsnorm(enable: bool, reload_user_impl: bool) -> str:
    del reload_user_impl  # The user module is reloaded on every execution.

    with _PATCH_LOCK:
        targets = list(_available_targets())
        if not targets:
            raise RuntimeError("No RMSNorm forward targets were found in comfy.ops")

        for target, attribute_name in targets:
            key = (_target_name(target), attribute_name)
            _ORIGINAL_FUNCTIONS.setdefault(key, getattr(target, attribute_name))

        if not enable:
            for target, attribute_name in targets:
                setattr(target, attribute_name, _ORIGINAL_FUNCTIONS[(_target_name(target), attribute_name)])
            logger.info("[baymax-zimage] Restored the default z-Image RMSnorm implementation")
            return "restored"

        user_module = _load_user_module()
        user_function = getattr(user_module, "apply_rmsnorm", None)
        if not callable(user_function):
            raise RuntimeError(f"{_USER_IMPL_PATH} must define a callable apply_rmsnorm function")

        for target, attribute_name in targets:
            setattr(target, attribute_name, _make_wrapper(_target_name(target), user_function))

        logger.info("[baymax-zimage] Installed user RMSnorm implementation from %s", _USER_IMPL_PATH)
        return "patched"


class BaymaxZImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "reload_user_impl": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "baymax"
    DESCRIPTION = (
        "Patch ComfyUI RMSNorm forward used by z-Image NextDiT with the implementation in "
        "custom_nodes/baymax-zimage/user_impl.py."
    )

    def patch_model(self, model, enable=True, reload_user_impl=True):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if diffusion_model is not None and diffusion_model.__class__.__name__ != "NextDiTPixelSpace":
            logger.warning(
                "[baymax-zimage] Received model type %s; patch applies globally to ComfyUI RMSNorm code paths",
                diffusion_model.__class__.__name__,
            )

        patched_model = model.clone()
        status = _patch_rmsnorm(enable=enable, reload_user_impl=reload_user_impl)

        transformer_options = patched_model.model_options.setdefault("transformer_options", {})
        transformer_options["baymax_zimage"] = {
            "status": status,
            "user_impl": str(_USER_IMPL_PATH),
        }
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "BaymaxZImage": BaymaxZImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaymaxZImage": "baymax-zimage",
}