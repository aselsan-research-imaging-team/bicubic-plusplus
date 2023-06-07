import numpy as np
import math
import cv2
import torch
from utils.imresize import imresize


def calc_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, noise=False, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    if noise:
        kernel = kernel + np.random.uniform(0, 0.25, (l, l)) * kernel
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, noise=False, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = calc_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, noise=noise, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, noise=False, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, noise=noise,
                                                  tensor=tensor)


def gaussian_noise(img, mean=0.0, sigma=0.2, tensor=False):
    if not tensor:
        g = np.random.normal(mean, sigma, img.shape)
        noisy_img = img + g
        return np.clip(noisy_img, np.min(img), np.max(img))
    else:
        noise = torch.FloatTensor(
            np.random.normal(loc=mean, scale=sigma, size=tensor.size())
        ).to(tensor.device)
        return torch.clamp(noise + tensor, min=0.0, max=1.0)


class Degrader(object):
    def __init__(self,
                 ds_rate=1,
                 enable_blur=False,
                 enable_img_noise=False,
                 enable_kernel_noise=False,
                 kernel_size=21,
                 rate_isotropic=1.0,
                 sig_min=0.2,
                 sig_max=2.6,
                 img_noise_level=0.2,
                 load_kernels_from_disc=False,
                 kernel_path=False):
        self.ds_rate = ds_rate
        self.enable_blur = enable_blur
        self.enable_img_noise = enable_img_noise
        self.enable_kernel_noise = enable_kernel_noise
        self.kernel_size = kernel_size
        self.rate_isotropic = rate_isotropic
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.img_noise_level = img_noise_level
        self.load_kernels_from_disc = load_kernels_from_disc
        self.kernel_path = kernel_path

    def __call__(self, data):
        deg_img, kernel = data

        deg_img = deg_img.astype(np.float32) / 255.
        if self.enable_blur:
            if not self.load_kernels_from_disc:
                #  Overwrite kernel with randomly generated one
                kernel = random_gaussian_kernel(l=self.kernel_size,
                                                sig_min=self.sig_min,
                                                sig_max=self.sig_max,
                                                rate_iso=self.rate_isotropic,
                                                scaling=self.ds_rate,
                                                noise=self.enable_kernel_noise)
            deg_img = cv2.filter2D(src=deg_img, ddepth=-1, kernel=kernel)

        if self.ds_rate != 1:
            deg_img = imresize(deg_img, 1 / self.ds_rate)

        if self.enable_img_noise:
            deg_img = gaussian_noise(deg_img, mean=0.0, sigma=self.img_noise_level)

        deg_img = (deg_img * 255).astype(np.uint8)
        return deg_img, kernel
