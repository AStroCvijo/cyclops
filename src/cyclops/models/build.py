"""Assemble the depth model (encoder + decoder) from an experiment config.

This is the one place approaches branch: the ResNet-50 baseline (approach 1) and
the frozen Stable Diffusion UNet encoder (approach 2) are wired up here; the other
frozen-encoder approaches plug in the same way.
"""

import torch.nn as nn

from cyclops.models.decoders.lightweight import LightweightDecoder
from cyclops.models.encoders.resnet50 import ResNet50Encoder
from cyclops.models.encoders.sd_unet import SDUNetEncoder


# Function for building an encoder from its config block
def build_encoder(enc_cfg):
    name = enc_cfg["name"]
    if name == "resnet50":
        return ResNet50Encoder(pretrained=enc_cfg.get("pretrained", True))
    if name == "sd_unet":
        return SDUNetEncoder(
            model_id=enc_cfg["model_id"],
            timestep=enc_cfg.get("timestep", 1),
            feature_blocks=enc_cfg["feature_blocks"],
            prompt=enc_cfg.get("prompt", ""),
        )
    raise ValueError(f"encoder {name!r} is not implemented yet")


# Function for building a decoder from its config block
def build_decoder(dec_cfg, in_channels, max_depth):
    name = dec_cfg["name"]
    if name == "lightweight":
        return LightweightDecoder(in_channels, dec_cfg["out_channels"], max_depth)
    raise ValueError(f"decoder {name!r} is not implemented yet")


class DepthModel(nn.Module):
    # Function for holding the encoder and decoder together
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # Function for predicting a depth map from an image
    def forward(self, image):
        feats = self.encoder(image)
        return self.decoder(feats, image.shape[-2:])


# Function for building the full depth model from the experiment config
def build_model(cfg):
    encoder = build_encoder(cfg["model"]["encoder"])
    decoder = build_decoder(
        cfg["model"]["decoder"],
        in_channels=encoder.out_channels,
        max_depth=cfg["dataset"]["depth"]["max"],
    )
    return DepthModel(encoder, decoder)
