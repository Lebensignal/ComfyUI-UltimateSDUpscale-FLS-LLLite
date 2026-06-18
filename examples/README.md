# Example Workflow

This folder contains a minimal API-format workflow that demonstrates where the FLS + LLLite upscale node fits in an Anima-style generation graph.

Files:

- `anima-fls-lllite-basic/workflow.json`
- `anima-fls-lllite-basic/schema.json`

The workflow intentionally uses placeholder model and LoRA filenames. Replace every `example_*.safetensors` and `example_*.pth` value with filenames from your own ComfyUI installation before running it.

The example assumes these node families are available:

- Anima model loader / TeaCache nodes
- BSS FLSampler node
- this repository's `UltimateSDUpscaleFLSLLLite`
- standard ComfyUI CLIP, VAE, LoRA, upscale model, and save image nodes

No private model names, LoRA names, output paths, API keys, or deployment settings are included.

