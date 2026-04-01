from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import threading
from pathlib import Path
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


def _load_user_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("baymax_zimage_user_impl", _USER_IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load user implementation from {_USER_IMPL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_user_impl(user_function: Callable, original_function: Callable, module_self, x):
    signature = inspect.signature(user_function)
    parameters = signature.parameters
    parameter_values = parameters.values()
    positional_slots = sum(
        1
        for parameter in parameter_values
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameter_values)
    has_varkw = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameter_values)

    weight = getattr(module_self, "weight", None)
    eps = getattr(module_self, "eps", 1e-6)

    def fallback(x_in, weight_in=None):
        del weight_in
        return original_function(module_self, x_in)

    args = [x, weight]
    kwargs = {}

    if "original_apply_rmsnorm" in parameters or has_varkw:
        kwargs["original_apply_rmsnorm"] = fallback
    if "eps" in parameters or has_varkw:
        kwargs["eps"] = eps
    if "module" in parameters or has_varkw:
        kwargs["module"] = module_self

    if positional_slots >= 3 and "original_apply_rmsnorm" not in kwargs:
        args.append(fallback)

    if positional_slots < len(args) and not has_varargs:
        args = args[:max(positional_slots, 0)]

    return user_function(*args, **kwargs)


def _make_wrapper(module_name: str, user_function: Callable) -> Callable:
    original_function = _ORIGINAL_FUNCTIONS[(module_name, "forward")]

    def patched_forward(self, x):
        return _call_user_impl(user_function, original_function, self, x)

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