import yaml
import numpy as np
import cv2
import torch
import math
import torch.nn as nn


class Bicubic(nn.Module):
    def __init__(self, scale=4, use_gpu=True):
        super(Bicubic, self).__init__()
        self.sc = scale
        self.use_gpu = use_gpu

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input):
        scale = self.sc
        [b, c, h, w] = input.shape
        output_size = [b, c, int(h * scale), int(w * scale)]

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)

        weight0 = np.asarray(weight0[0], dtype=np.float32)
        if self.use_gpu == True:
            weight0 = torch.from_numpy(weight0).cuda()
        else:
            weight0 = torch.from_numpy(weight0)

        indice0 = np.asarray(indice0[0], dtype=np.float32)
        if self.use_gpu == True:
            indice0 = torch.from_numpy(indice0).cuda().long()
        else:
            indice0 = torch.from_numpy(indice0).long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        weight1 = np.asarray(weight1[0], dtype=np.float32)
        if self.use_gpu == True:
            weight1 = torch.from_numpy(weight1).cuda()
        else:
            weight1 = torch.from_numpy(weight1)

        indice1 = np.asarray(indice1[0], dtype=np.float32)
        if self.use_gpu == True:
            indice1 = torch.from_numpy(indice1).cuda().long()
        else:
            indice1 = torch.from_numpy(indice1).long()

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.round(255 * torch.sum(out, dim=3).permute(0, 1, 3, 2)) / 255

        return out


def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx2 * absx

    weight = (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (
            absx <= 2)).type_as(
        absx))
    return weight


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round
    is_numpy = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.transpose(2, 0, 1))
        is_numpy = True
    device = img.device
    # device = torch.device("cuda")

    is_batch = True
    if len(img.shape) == 3:  # C, H, W
        img = img[None]
        is_batch = False

    B, in_C, in_H, in_W = img.size()
    img = img.view(-1, in_H, in_W)
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    # print(weights_H.device, indices_H.device, device)
    weights_H, indices_H = weights_H.to(device), indices_H.to(device)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W = weights_W.to(device), indices_W.to(device)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(B * in_C, in_H + sym_len_Hs + sym_len_He, in_W).to(device)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(B * in_C, out_H, in_W).to(device)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[:, i, :] = (img_aug[:, idx:idx + kernel_width, :].transpose(1, 2).matmul(
            weights_H[i][None, :, None].repeat(B * in_C, 1, 1))).squeeze()

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(B * in_C, out_H, in_W + sym_len_Ws + sym_len_We).to(device)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(B * in_C, out_H, out_W).to(device)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, :, i] = (out_1_aug[:, :, idx:idx + kernel_width].matmul(
            weights_W[i][None, :, None].repeat(B * in_C, 1, 1))).squeeze()

    out_2 = out_2.contiguous().view(B, in_C, out_H, out_W)
    if not is_batch:
        out_2 = out_2[0]
    return out_2.cpu().numpy().transpose(1, 2, 0) if is_numpy else out_2


def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, :]
    return ims


def imread(filename, dtype=np.uint8):
    img = cv2.imread(filename)
    img = img.astype(dtype)
    return img


class Imread_Modcrop:
    def __init__(self, mod):
        self.mod = mod

    def __call__(self, filename, dtype=np.uint8):
        img = imread(filename)
        img = modcrop(img, self.mod)
        return img


def shave(image, scale):
    return image[:, :, scale:-scale, scale:-scale]


def rgb_to_ycbcr(input):
    # input is mini-batch N x 3 x H x W of an RGB image
    return ((input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :,
                                                                        :] * 24.966 + 16) / 255).unsqueeze(1)


if __name__ == '__main__':
    pass
