"""Assemble the depth model (encoder + decoder) from an experiment config.

This is the one place approaches branch: the ResNet-50 baseline (approach 1), the
frozen Stable Diffusion UNet encoder (approach 2), and the frozen I-JEPA encoder +
DPT decoder (approach 3) are wired up here; the remaining approaches plug in the
same way.
"""

import torch.nn as nn

from cyclops.models.decoders.dpt import DPTDecoder
from cyclops.models.decoders.lightweight import LightweightDecoder
from cyclops.models.encoders.ijepa import IjepaEncoder
from cyclops.models.encoders.resnet50 import ResNet50Encoder
from cyclops.models.encoders.sd_unet import SDUNetEncoder
from cyclops.models.fusion import ConcatFusion, CrossAttentionFusion


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
    if name == "ijepa":
        return IjepaEncoder(
            model_name=enc_cfg["model_name"],
            tap_layers=enc_cfg["tap_layers"],
            image_size=enc_cfg["image_size"],
        )
    raise ValueError(f"encoder {name!r} is not implemented yet")


# Function for building a decoder from its config block
def build_decoder(dec_cfg, in_channels, max_depth):
    name = dec_cfg["name"]
    if name == "lightweight":
        return LightweightDecoder(in_channels, dec_cfg["out_channels"], max_depth)
    if name == "dpt":
        return DPTDecoder(
            in_channels, dec_cfg["reassemble_channels"], dec_cfg["out_channels"], max_depth
        )
    raise ValueError(f"decoder {name!r} is not implemented yet")


# Function for building a fusion module from its config block and the two encoders' channels
def build_fusion(fus_cfg, channels_a, channels_b):
    method = fus_cfg["method"]
    if method == "concat":
        return ConcatFusion(channels_a, channels_b)
    if method == "cross_attention":
        return CrossAttentionFusion(
            channels_a, channels_b, dim=fus_cfg.get("dim", 256), heads=fus_cfg.get("heads", 8)
        )
    raise ValueError(f"fusion {method!r} is not implemented yet")


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


class FusionModel(nn.Module):
    # Function for holding two frozen encoders, a fusion module, and a decoder
    def __init__(self, encoders, fusion, decoder):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fusion = fusion
        self.decoder = decoder

    # Function for predicting a depth map by fusing both encoders' features
    def forward(self, image):
        feats_a = self.encoders[0](image)
        feats_b = self.encoders[1](image)
        fused = self.fusion(feats_a, feats_b)
        return self.decoder(fused, image.shape[-2:])


# Function for building the full depth model from the experiment config
def build_model(cfg):
    m = cfg["model"]
    max_depth = cfg["dataset"]["depth"]["max"]

    if "encoders" in m:                       # approach 4: fuse two frozen encoders
        encoders = [build_encoder(e) for e in m["encoders"]]
        fusion = build_fusion(m["fusion"], encoders[0].out_channels, encoders[1].out_channels)
        decoder = build_decoder(m["decoder"], in_channels=fusion.out_channels, max_depth=max_depth)
        return FusionModel(encoders, fusion, decoder)

    encoder = build_encoder(m["encoder"])
    decoder = build_decoder(m["decoder"], in_channels=encoder.out_channels, max_depth=max_depth)
    return DepthModel(encoder, decoder)
