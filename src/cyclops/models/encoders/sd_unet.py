"""Frozen Stable Diffusion UNet encoder: SD features -> multi-scale maps.

Stable Diffusion is an image generator, but its UNet's internal activations are
rich general-purpose visual features. We run SD frozen: encode the RGB image to a
VAE latent, push it once through the UNet at a fixed low `timestep` with an
empty-prompt text embedding, and tap a few UNet blocks. Forward hooks capture
each tapped block's output; we return them as feature maps ordered fine -> coarse
so the lightweight decoder can consume them exactly like the ResNet-50 features.
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from cyclops.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from cyclops.models.encoders.base import Encoder


class SDUNetEncoder(Encoder):
    # Function for loading a frozen SD pipeline and wiring up the feature taps
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", timestep=1,
                 feature_blocks=("up_1", "up_2", "up_3", "mid"), prompt=""):
        super().__init__()
        self.timestep = int(timestep)
        self.feature_blocks = list(feature_blocks)

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

        for module in (self.vae, self.unet, text_encoder):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

        # Precompute the (empty) prompt embedding once; SD conditioning is constant.
        tokens = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            embeds = text_encoder(tokens.input_ids)[0]     # (1, 77, cross_attention_dim)
        self.register_buffer("text_embeds", embeds)

        # Reverse the dataset's ImageNet normalization before the VAE (SD wants [-1, 1]).
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # Hook the requested blocks so a single UNet pass records their outputs.
        self._features = {}
        for name in self.feature_blocks:
            self._resolve_block(name).register_forward_hook(self._make_hook(name))

        # Probe once to learn each tap's channel count and resolution, then fix the
        # fine -> coarse order and out_channels the decoder relies on.
        with torch.no_grad():
            latent_ch = self.vae.config.latent_channels
            t = torch.zeros(1, dtype=torch.long)
            self._features = {}
            self.unet(torch.zeros(1, latent_ch, 32, 32), t, encoder_hidden_states=self.text_embeds)
        sizes = {n: f.shape[-2] * f.shape[-1] for n, f in self._features.items()}
        self._order = sorted(self._features, key=lambda n: sizes[n], reverse=True)   # fine -> coarse
        self.out_channels = [self._features[n].shape[1] for n in self._order]

    # Function for mapping a config block name ("up_1", "mid", ...) to its module
    def _resolve_block(self, name):
        if name == "mid":
            return self.unet.mid_block
        kind, idx = name.rsplit("_", 1)
        blocks = {"up": self.unet.up_blocks, "down": self.unet.down_blocks}[kind]
        return blocks[int(idx) - 1]

    # Function for building a forward hook that stores a block's output feature map
    def _make_hook(self, name):
        def hook(module, inputs, output):
            self._features[name] = output[0] if isinstance(output, tuple) else output
        return hook

    # Function for keeping the frozen SD submodules in eval mode even while training
    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        self.unet.eval()
        return self

    # Function for returning the tapped SD feature maps (fine -> coarse)
    def forward(self, x):
        # Frozen SD runs in fp32 even under the trainer's AMP autocast: the SD 2.1
        # VAE is unstable in fp16 (NaNs), and the features are detached anyway.
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float() * self.imagenet_std + self.imagenet_mean   # ImageNet-normalized -> [0, 1]
            x = 2.0 * x - 1.0                                        # -> [-1, 1] for the VAE
            latent = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor
            t = torch.full((x.shape[0],), self.timestep, device=x.device, dtype=torch.long)
            embeds = self.text_embeds.expand(x.shape[0], -1, -1)
            self._features = {}
            self.unet(latent, t, encoder_hidden_states=embeds)
        return [self._features[name] for name in self._order]
