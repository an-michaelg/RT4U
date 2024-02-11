from PIL import ImageOps, ImageFilter
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

IMG_MEAN = {'natural':[0.485, 0.456, 0.406], 
            'AS':[0.099, 0.099, 0.099], 
            'unit':[0.0, 0.0, 0.0]}
IMG_STD = {'natural':[0.229, 0.224, 0.225], 
           'AS':[0.171, 0.171, 0.171], 
           'unit':[1.0, 1.0, 1.0]}

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.4 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
            
class IncreaseSharp(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sharp = np.random.rand() * 3.9 + 1.1
            return TF.adjust_sharpness(img, sharp)
        else:
            return img

class AdaptiveGamma(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img):
        if np.random.rand() < self.p:
            gamma = np.log(0.5*255)/np.log(np.mean(img))
            return TF.adjust_gamma(img, 1/gamma)
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
            
def pil_transform():
    return transforms.ToPILImage()
    
def image_space_transform(img_resolution, rotate_degrees, crop_scale_min, 
                          aspect_min=0.8, aspect_max=1.2, p_hflip=0.5):
    transform = transforms.Compose(
        [
            transforms.RandomRotation(rotate_degrees),
            transforms.RandomResizedCrop(img_resolution, 
                                         scale=(crop_scale_min,1.0), 
                                         ratio=(aspect_min, aspect_max)
                                        ),
            transforms.RandomHorizontalFlip(p=p_hflip),
        ]
    )
    return transform

# superset of possible random transformations on natural images
def image_color_transform(p_autocontrast=0.5, p_jitter=0.8,
                          p_gray=0.2, p_gblur=0.5, p_solar=0.0,
                          img_mean = IMG_MEAN['natural'], img_std = IMG_STD['natural'],
                          mode='color'):
    if mode == 'color':
        transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=p_jitter,
                ),
                transforms.RandomGrayscale(p=p_gray),
                GaussianBlur(p=p_gblur),
                Solarization(p=p_solar),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )
    elif mode == 'grayscale':
        transform = transforms.Compose(
            [
                AdaptiveGamma(p=1.0),
                #transforms.RandomAutocontrast(p=p_autocontrast),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.5, saturation=0.4, hue=0.0
                        )
                    ],
                    p=p_jitter,
                ),
                IncreaseSharp(p=p_gblur),
                GaussianBlur(p=p_gblur),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )
        
    return transform

# return two augmented versions of the original image
class DualAugmentTransform(object):
    def __init__(self, resolution, SSL_mode, norm_setting='natural'):
        img_mean = IMG_MEAN[norm_setting]
        img_std = IMG_STD[norm_setting]
        self.SSL_mode = SSL_mode
        
        if norm_setting == 'natural':
            rotate = 10
            crop_scale_min = 0.7
            aspect_min = 0.75
            aspect_max = 1.333
            p_hflip = 0.5
            p_autocontrast = 0.5
            p_jitter = 0.8
            p_gray = 0.2
            p_gblur = 0.5
            p_solar = 0.2
            mode = 'color'
        elif norm_setting == 'AS':
            rotate = 10
            crop_scale_min = 0.7
            aspect_min = 0.8
            aspect_max = 1.2
            p_hflip = 0.0
            p_autocontrast = 0.5
            p_jitter = 1.0
            p_gray = 0.0
            p_gblur = 0.5
            p_solar = 0.0
            mode = 'grayscale'
        elif norm_setting == 'unit':
            rotate = 0.0
            crop_scale_min = 0.99
            aspect_min = 0.99
            aspect_max = 1.01
            p_hflip = 0.0
            p_autocontrast = 0.0
            p_jitter = 0.0
            p_gray = 0.0
            p_gblur = 0.0
            p_solar = 0.0
            
        self.space_transform = image_space_transform(resolution, rotate, 
                                                     crop_scale_min, aspect_min, aspect_max, p_hflip)
        self.color_transform = image_color_transform(p_autocontrast, p_jitter, 
                                                     p_gray, p_gblur, p_solar, img_mean, img_std, mode)

    def __call__(self, sample):
        x = self.space_transform(sample)
        x1 = self.color_transform(x)
        if self.SSL_mode == 'patch':
            x2 = self.color_transform(x)
        elif self.SSL_mode == 'image':
            x2 = self.color_transform(self.space_transform(sample))
        else:
            raise ValueError(f'SSL_mode should be image/patch, got {self.SSL_mode}')
        return {'x':x1, 'x_aug':x2}
        
# minimalist transform of the original image
class ImageIdentityTransform(object):
    def __init__(self, resolution, norm_setting='natural'):
        img_mean = IMG_MEAN[norm_setting]
        img_std = IMG_STD[norm_setting]
        self.space_transform = image_space_transform(resolution, 
                                                     rotate_degrees=0.0,
                                                     crop_scale_min=0.99, 
                                                     aspect_min=0.99, 
                                                     aspect_max=1.01, 
                                                     p_hflip=0.0)
        self.color_transform = image_color_transform(p_autocontrast=0.0, 
                                                     p_jitter=0.0,
                                                     p_gray=0.0, 
                                                     p_gblur=0.0, 
                                                     p_solar=0.0, 
                                                     img_mean=img_mean, 
                                                     img_std=img_std,
                                                     mode='color')

    def __call__(self, sample):
        x1 = self.color_transform(self.space_transform(sample))
        return {'x':x1}

class RandomRotateVideo(object):

    def __init__(self, degrees, expand=False, center=None, fill=0):
        if type(degrees) == int or type(degrees) == float:
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees_low = -degrees
            self.degrees_high = degrees
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be (min_angle, max_angle).")
            self.degrees_low = degrees[0]
            self.degrees_high = degrees[1]

        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, x):
        # this is kinda sketchy but I didn't want to write a method
        # that generalizes to PIL video
        assert torch.is_tensor(x)
        c, t, h, w = x.shape
        angle = random.uniform(self.degrees_low, self.degrees_high)
        for i in range(t):
            x[:, i, :, :] = TF.rotate(x[:, i, :, :], angle, expand=self.expand, center=self.center, fill=self.fill)
            #  the same transform on the whole "batch" (batch = all the extraacted frames of the cine, be it 1 or more)
        return x