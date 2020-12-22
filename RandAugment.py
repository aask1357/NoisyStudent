import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):  # [0, 256] -> [256, 0]
    assert 0 <= v <= 256
    v = 256 - int(v)
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [0, 4] -> [8, 4]
    assert 0 <= v <= 4
    v = 8 - round(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.0, 0.9] -> [0.1,1.9]
    assert 0.0 <= v <= 0.9
    v = 1.0 + v if random.random() > 0.5 else 1.0 - v
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.0, 0.9] -> [0.1,1.9]
    assert 0.0 <= v <= 0.9
    v = 1.0 + v if random.random() > 0.5 else 1.0 - v
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.0, 0.9] -> [0.1,1.9]
    assert 0.0 <= v <= 0.9
    v = 1.0 + v if random.random() > 0.5 else 1.0 - v
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.0, 0.9] -> [0.1,1.9]
    assert 0.0 <= v <= 0.9
    v = 1.0 + v if random.random() > 0.5 else 1.0 - v
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Identity(img, v):
    return img


def augment_list():
    l = [
        (Identity, 0., 1.),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Rotate, 0, 30),
        (Solarize, 0, 256),
        (Color, 0., 0.9),
        (Posterize, 0., 4.),
        (Contrast, 0., 0.9),
        (Brightness, 0., 0.9),
        (Sharpness, 0., 0.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (TranslateXabs, 0., 10),
        (TranslateYabs, 0., 10),
    ]

    return l


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30.) * float(maxval - minval) + minval
            img = op(img, val)

        return img

class RandAugmentDebug(RandAugment):
    def __init__(self, n, m):
        super().__init__(n, m)
        
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            print(op.__name__)
            plt.figure()
            plt.imshow(np.asarray(img))
            val = (float(self.m) / 30.) * float(maxval - minval) + minval
            img = op(img, val)

        return img
