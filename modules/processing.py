from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import math
import importlib.util
import os
import sys
from comfy_extras.nodes_custom_sampler import SamplerCustom
from utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond
from modules import shared
from tqdm import tqdm
import comfy
import comfy.sample
import comfy.model_management
import latent_preview
from enum import Enum
import json

LLLITE_NONE = "None"
ANIMA_LLLITE_NODE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "ComfyUI-Anima-LLLite")
)
_ANIMA_LLLITE_APPLY = None


def _load_anima_lllite_apply():
    global _ANIMA_LLLITE_APPLY
    if _ANIMA_LLLITE_APPLY is not None:
        return _ANIMA_LLLITE_APPLY

    init_path = os.path.join(ANIMA_LLLITE_NODE_DIR, "__init__.py")
    if not os.path.exists(init_path):
        raise FileNotFoundError(
            "ComfyUI-Anima-LLLite is not installed. Expected node package: "
            f"{ANIMA_LLLITE_NODE_DIR}"
        )

    package_name = "_usdu_fls_anima_lllite"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[ANIMA_LLLITE_NODE_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    _ANIMA_LLLITE_APPLY = module.NODE_CLASS_MAPPINGS["AnimaLLLiteApply"]()
    return _ANIMA_LLLITE_APPLY


def _load_comfy_nodes_module():
    module = sys.modules.get("nodes")
    if module is not None and hasattr(module, "common_ksampler"):
        return module

    comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    nodes_path = os.path.join(comfy_root, "nodes.py")
    spec = importlib.util.spec_from_file_location("_comfy_root_nodes_for_usdu_fls", nodes_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_comfy_nodes = _load_comfy_nodes_module()
common_ksampler = _comfy_nodes.common_ksampler
VAEEncode = _comfy_nodes.VAEEncode
VAEDecode = _comfy_nodes.VAEDecode
VAEDecodeTiled = _comfy_nodes.VAEDecodeTiled

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

# Taken from the USDU script
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
        fovea_strength=5.0,
        sharpness=1.0,
        mask_inertia=0.85,
        lllite_model_name=None,
        lllite_strength=0.0,
        lllite_start_percent=0.0,
        lllite_end_percent=0.9,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height
        self.rows = math.ceil(self.height / tile_height)
        self.cols = math.ceil(self.width / tile_width)

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas
        self.fovea_strength = fovea_strength
        self.sharpness = sharpness
        self.mask_inertia = mask_inertia
        self.lllite_model_name = lllite_model_name
        self.lllite_strength = lllite_strength
        self.lllite_start_percent = lllite_start_percent
        self.lllite_end_percent = lllite_end_percent
        self.lllite_enabled = bool(
            lllite_model_name and lllite_model_name != LLLITE_NONE
            and lllite_strength is not None and float(lllite_strength) != 0.0
        )

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            print("[USDU] Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            print("[USDU] Using tiled decode")

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}

        # Load config file for USDU
        config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        if comfy.utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy.utils.PROGRESS_BAR_ENABLED = config.get('per_tile_progress', True)
            self.tiles = 0
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)
            self.pbar = None
            # self.pbar = tqdm(total=self.tiles, desc='USDU') # Creating the pbar here will cause an empty progress bar to be displayed

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled:
            comfy.utils.PROGRESS_BAR_ENABLED = True
    
class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def sample_fls(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, denoise, fovea_strength, sharpness, mask_inertia):
    latent_dict = latent_image.copy()
    latent = latent_dict["samples"]
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    latent_dict["samples"] = latent

    noise_mask = latent_dict.get("noise_mask")
    device = comfy.model_management.get_torch_device()
    noise = comfy.sample.prepare_noise(latent, seed)
    preview_callback = latent_preview.prepare_callback(model, steps)

    previous_pred = None
    accumulated_mask = torch.zeros(
        (latent.shape[0], 1, latent.shape[-2], latent.shape[-1]), device=device
    )
    momentum_mask = None

    def fls_callback(step, x0, x, total_steps):
        nonlocal previous_pred, accumulated_mask, momentum_mask

        if step < total_steps * 0.10 or previous_pred is None:
            previous_pred = x0
            preview_callback(step, x0, x, total_steps)
            return

        delta = torch.abs(x0 - previous_pred)
        delta_map = torch.mean(delta, dim=1, keepdim=True)
        orig_shape = delta_map.shape
        if len(orig_shape) == 5:
            delta_map = delta_map.reshape(-1, 1, orig_shape[3], orig_shape[4])

        delta_smooth = F.avg_pool2d(delta_map, kernel_size=5, stride=1, padding=2)
        if len(orig_shape) == 5:
            delta_smooth = delta_smooth.view(orig_shape)

        mean_val = delta_smooth.mean()
        std_val = delta_smooth.std()
        threshold = mean_val + (std_val * 0.5)
        current_mask = torch.sigmoid((delta_smooth - threshold) / (std_val + 1e-6) * 2.0)
        current_mask = torch.where(
            current_mask < 0.2,
            torch.tensor(0.0, device=x.device, dtype=current_mask.dtype),
            current_mask,
        )

        if momentum_mask is None:
            momentum_mask = current_mask
        else:
            momentum_mask = momentum_mask * mask_inertia + current_mask * (1.0 - mask_inertia)

        active_mask = momentum_mask
        if accumulated_mask.device != active_mask.device:
            accumulated_mask = accumulated_mask.to(active_mask.device)
        accumulated_mask += active_mask.squeeze(2) if len(active_mask.shape) == 5 else active_mask

        progress = step / total_steps
        decay = 1.0 - progress

        if sharpness > 0:
            x0_4d = (
                x0.view(-1, x0.shape[-3], x0.shape[-2], x0.shape[-1])
                if len(x0.shape) == 5
                else x0
            )
            blurred_x0 = F.avg_pool2d(x0_4d, kernel_size=3, stride=1, padding=1)
            if len(x0.shape) == 5:
                blurred_x0 = blurred_x0.view(x0.shape)
            x += (x0 - blurred_x0) * active_mask * (sharpness * 0.1 * decay)

        if fovea_strength > 0:
            perturbation = torch.randn_like(x) * active_mask * (fovea_strength * 0.02 * decay)
            x += torch.clamp(perturbation, -0.15, 0.15)

        previous_pred = x0
        preview_callback(step, x0, x, total_steps)

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=True,
        noise_mask=noise_mask,
        callback=fls_callback,
        disable_pbar=False,
        seed=seed,
    )

    out_latent = latent_dict.copy()
    out_latent.pop("downscale_ratio_spacial", None)
    out_latent["samples"] = samples
    return out_latent


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise,
           custom_sampler, custom_sigmas, fovea_strength=None, sharpness=None, mask_inertia=None):
    # Choose way to sample based on given inputs

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        custom_sample = SamplerCustom()
        (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        return samples

    if fovea_strength is not None and sharpness is not None and mask_inertia is not None:
        return sample_fls(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
            latent, denoise, fovea_strength, sharpness, mask_inertia
        )

    # Default
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


def apply_lllite_tile_patch(p: StableDiffusionProcessing, cond_image):
    if not p.lllite_enabled:
        return p.model

    import folder_paths

    weights_path = folder_paths.get_full_path("controlnet", p.lllite_model_name)
    if weights_path is None or not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Anima LLLite model file not found. Put "
            f"{p.lllite_model_name} in ComfyUI/models/controlnet"
        )

    applier = _load_anima_lllite_apply()
    (model_lllite,) = applier.apply(
        p.model,
        p.lllite_model_name,
        cond_image,
        p.lllite_strength,
        p.lllite_start_percent,
        p.lllite_end_percent,
        preserve_wrapper=True,
        mask=None,
    )
    return model_lllite


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height
    else:
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in shared.batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)
    sample_model = apply_lllite_tile_patch(p, batched_tiles)

    # Generate samples
    samples = sample(sample_model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                     negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas,
                     p.fovea_strength, p.sharpness, p.mask_inertia)

    # Update the progress bar
    if p.progress_bar_enabled:
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, None)
    return processed
