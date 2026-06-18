# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import logging
import os
import torch
import comfy
import folder_paths
from usdu_patch import usdu
from utils import tensor_to_pil, pil_to_tensor
from modules.processing import StableDiffusionProcessing
import modules.shared as shared
from modules.upscaler import UpscalerData

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}

LLLITE_NONE = "None"
EXPECTED_LLLITE_MODEL = "anima_tiled_lllite_v1.safetensors"


def lllite_model_dir():
    comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(comfy_dir, "models", "controlnet")


def get_lllite_model_names():
    try:
        names = folder_paths.get_filename_list("controlnet")
    except Exception:
        model_dir = lllite_model_dir()
        names = []
        if os.path.isdir(model_dir):
            names = [
                name for name in os.listdir(model_dir)
                if os.path.isfile(os.path.join(model_dir, name)) and name != "put_models_here.txt"
            ]
    if EXPECTED_LLLITE_MODEL not in names:
        names.insert(0, EXPECTED_LLLITE_MODEL)
    return [LLLITE_NONE] + sorted(names, key=lambda x: (x != EXPECTED_LLLITE_MODEL, x.lower()))


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE",)),
        # Sampling Params
        ("model", ("MODEL",)),
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("fovea_strength", ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"})),
        ("sharpness", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, "display": "slider"})),
        ("mask_inertia", ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01, "display": "slider"})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]

    optional = []

    return required, optional


def add_lllite_inputs(required: list):
    insert_at = len(required)
    for i, (name, _) in enumerate(required):
        if name == "mask_inertia":
            insert_at = i + 1
            break

    lllite_inputs = [
        ("lllite_model_name", (get_lllite_model_names(), {"default": EXPECTED_LLLITE_MODEL})),
        ("lllite_strength", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"})),
        ("lllite_start_percent", ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001})),
        ("lllite_end_percent", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.001})),
    ]
    for offset, item in enumerate(lllite_inputs):
        required.insert(insert_at + offset, item)


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                fovea_strength, sharpness, mask_inertia,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, 
                custom_sampler=None, custom_sigmas=None,
                lllite_model_name=None, lllite_strength=0.0,
                lllite_start_percent=0.0, lllite_end_percent=0.9):
        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.upscale_by = upscale_by
        self.seam_fix_mask_blur = seam_fix_mask_blur

        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image), model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            custom_sampler, custom_sigmas, fovea_strength, sharpness, mask_inertia,
            lllite_model_name, lllite_strength, lllite_start_percent, lllite_end_percent,
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            #
            # Running the script
            #
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=self.tile_width, tile_height=self.tile_height,
                               mask_blur=self.mask_blur, padding=self.tile_padding, seams_fix_width=self.seam_fix_width,
                               seams_fix_denoise=self.seam_fix_denoise, seams_fix_padding=self.seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[self.mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=self.seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=self.upscale_by)

            # Return the resulting images
            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)
        finally:
            # Restore the original logging level
            logger.setLevel(old_level)


class UltimateSDUpscaleFLSLLLite(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        add_lllite_inputs(required)
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                fovea_strength, sharpness, mask_inertia,
                lllite_model_name, lllite_strength, lllite_start_percent, lllite_end_percent,
                upscale_model, mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                custom_sampler=None, custom_sigmas=None):
        return super().upscale(
            image=image,
            model=model,
            positive=positive,
            negative=negative,
            vae=vae,
            upscale_by=upscale_by,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            upscale_model=upscale_model,
            fovea_strength=fovea_strength,
            sharpness=sharpness,
            mask_inertia=mask_inertia,
            mode_type=mode_type,
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=mask_blur,
            tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode,
            seam_fix_denoise=seam_fix_denoise,
            seam_fix_mask_blur=seam_fix_mask_blur,
            seam_fix_width=seam_fix_width,
            seam_fix_padding=seam_fix_padding,
            force_uniform_tiles=force_uniform_tiles,
            tiled_decode=tiled_decode,
            custom_sampler=custom_sampler,
            custom_sigmas=custom_sigmas,
            lllite_model_name=lllite_model_name,
            lllite_strength=lllite_strength,
            lllite_start_percent=lllite_start_percent,
            lllite_end_percent=lllite_end_percent,
        )


class UltimateSDUpscaleNoUpscale(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscaled_image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):
        upscale_by = 1.0
        return super().upscale(upscaled_image, model, positive, negative, vae, upscale_by, seed,
                               steps, cfg, sampler_name, scheduler, denoise, None,
                               mode_type, tile_width, tile_height, mask_blur, tile_padding,
                               seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                               seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode)
    
class UltimateSDUpscaleCustomSample(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        optional.append(("upscale_model", ("UPSCALE_MODEL",)))
        optional.append(("custom_sampler", ("SAMPLER",)))
        optional.append(("custom_sigmas", ("SIGMAS",)))
        return prepare_inputs(required, optional)
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                upscale_model=None,
                custom_sampler=None, custom_sigmas=None):
        return super().upscale(image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode,
                custom_sampler, custom_sigmas)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleFLS": UltimateSDUpscale,
    "UltimateSDUpscaleFLSLLLite": UltimateSDUpscaleFLSLLLite,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleFLS": "Ultimate SD Upscale (FLS Accelerated)",
    "UltimateSDUpscaleFLSLLLite": "Ultimate SD Upscale (FLS + LLLite Tile Repair)",
}
