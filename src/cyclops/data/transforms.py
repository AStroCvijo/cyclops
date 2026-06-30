import numpy as np
import torch

# ImageNet statistics — the pretrained encoders (ResNet-50, ViT) expect this.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Function for turning an HxWx3 float image (0..1) into a normalized (3,H,W) tensor
def to_tensor(image):
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


# Function for light training augmentation applied jointly to image and depth
def augment(image, depth):
    # random horizontal flip — the same flip must apply to both image and depth
    if np.random.rand() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        depth = np.ascontiguousarray(depth[:, ::-1])
    # small brightness jitter on the RGB only (depth is geometry, leave it)
    if np.random.rand() < 0.5:
        image = np.clip(image * np.random.uniform(0.9, 1.1), 0.0, 1.0)
    return image, depth
