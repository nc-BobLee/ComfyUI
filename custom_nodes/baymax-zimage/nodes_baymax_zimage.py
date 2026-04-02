from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
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


def _available_targets(candidates: Iterable[Tuple[object, str]]) -> Tuple[Tuple[object, str], ...]:
    available = []
    for target, attribute_name in candidates:
        if hasattr(target, attribute_name):
            available.append((target, attribute_name))
        else:
            logger.warning(
                "[baymax-zimage] Skipping %s.%s because it does not exist",
                _target_name(target),
                attribute_name,
            )
    return tuple(available)


def _rmsnorm_target_modules() -> Iterable[Tuple[object, str]]:
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


def _rope_target_modules() -> Iterable[Tuple[object, str]]:
    # Patch both the source function in flux.math and the bound symbol imported
    # by lumina.model so z-Image paths pick up the custom implementation.
    return (
        (importlib.import_module("comfy.ldm.flux.math"), "apply_rope"),
        (importlib.import_module("comfy.ldm.lumina.model"), "apply_rope"),
    )


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


def _make_wrapper(user_function: Callable) -> Callable:
    # patched_forward calls apply_rmsnorm(x, weight, eps=eps) directly.
    # No dynamic Python in the hot path — Dynamo can trace straight through.
    def patched_forward(self, x):
        eps = self.eps if self.eps is not None else 1e-6
        return user_function(x, self.weight, eps=eps)

    patched_forward.__name__ = "forward"
    patched_forward.__module__ = __name__
    return patched_forward


def _make_rope_wrapper(user_function: Callable) -> Callable:
    def patched_apply_rope(xq, xk, freqs_cis):
        return user_function(xq, xk, freqs_cis)

    patched_apply_rope.__name__ = "apply_rope"
    patched_apply_rope.__module__ = __name__
    return patched_apply_rope


def _patch_baymax(enable: bool) -> Dict[str, str]:
    with _PATCH_LOCK:
        rmsnorm_targets = list(_available_targets(_rmsnorm_target_modules()))
        rope_targets = list(_available_targets(_rope_target_modules()))

        if not rmsnorm_targets:
            raise RuntimeError("No RMSNorm forward targets were found in comfy.ops")
        if not rope_targets:
            raise RuntimeError("No apply_rope targets were found in comfy.ldm.flux/lumina")

        all_targets = rmsnorm_targets + rope_targets

        for target, attribute_name in all_targets:
            key = (_target_name(target), attribute_name)
            _ORIGINAL_FUNCTIONS.setdefault(key, getattr(target, attribute_name))

        if not enable:
            for target, attribute_name in all_targets:
                setattr(target, attribute_name, _ORIGINAL_FUNCTIONS[(_target_name(target), attribute_name)])
            logger.info("[baymax-zimage] Restored default z-Image RMSNorm and apply_rope implementations")
            return {"rmsnorm": "restored", "apply_rope": "restored"}

        user_module = _load_user_module()
        user_rmsnorm = getattr(user_module, "apply_rmsnorm", None)
        user_rope = getattr(user_module, "apply_rope", None)

        if not callable(user_rmsnorm):
            raise RuntimeError(f"{_USER_IMPL_PATH} must define a callable apply_rmsnorm function")
        if not callable(user_rope):
            raise RuntimeError(f"{_USER_IMPL_PATH} must define a callable apply_rope function")

        for target, attribute_name in rmsnorm_targets:
            setattr(target, attribute_name, _make_wrapper(user_rmsnorm))
        for target, attribute_name in rope_targets:
            setattr(target, attribute_name, _make_rope_wrapper(user_rope))

        logger.info("[baymax-zimage] Installed user RMSNorm + apply_rope implementations from %s", _USER_IMPL_PATH)
        return {"rmsnorm": "patched", "apply_rope": "patched"}


class BaymaxZImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "baymax"
    DESCRIPTION = (
        "Patch ComfyUI RMSNorm forward and flux/lumina apply_rope used by z-Image NextDiT with implementations in "
        "custom_nodes/baymax-zimage/user_impl.py."
    )

    def patch_model(self, model, enable=True):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if diffusion_model is not None and diffusion_model.__class__.__name__ != "NextDiTPixelSpace":
            logger.warning(
                "[baymax-zimage] Received model type %s; patch applies globally to ComfyUI RMSNorm code paths",
                diffusion_model.__class__.__name__,
            )

        patched_model = model.clone()
        status = _patch_baymax(enable=enable)

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