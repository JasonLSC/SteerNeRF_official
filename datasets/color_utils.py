import pdb

import cv2
from einops import rearrange
import imageio
import numpy as np


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img>limit, ((img+0.055)/1.055)**2.4, img/12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img>limit, 1.055*img**(1/2.4)-0.055, 12.92*img)
    img[img>1] = 1 # "clamp" tonemapper
    return img


# def read_image(img_path, img_wh, blend_a=True):
#     img = imageio.imread(img_path).astype(np.float32)/255.0
#     mask = np.copy(img[..., -1:0])
#     # img[..., :3] = srgb_to_linear(img[..., :3])
#     if img.shape[2] == 4: # blend A to RGB
#         if blend_a:
#             img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
#         else:
#             img = img[..., :3]*img[..., -1:]
#
#     img = cv2.resize(img, img_wh)
#     img = rearrange(img, 'h w c -> (h w) c')
#
#     mask = cv2.resize(mask, img_wh)
#     mask = rearrange(mask, 'h w c -> (h w) c')
#
#     return img, mask

def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32)/255.0

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4: # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img

def read_mask(img_path, img_wh):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    mask = img[..., -1:] # object region mask

    mask = cv2.resize(mask, img_wh)
    mask = rearrange(mask, 'h w -> (h w) ')

    return mask
