from __future__ import division

import logging
import os
import torch.nn as nn
import cv2
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label + 1e-5)  #可能会除0
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def calc_psnr(img1, img2, data_range=255):
    img1 = img1.clamp(0., 1.).mul_(data_range).cpu().numpy()
    img2 = img2.clamp(0., 1.).mul_(data_range).cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close

class CutBlur:
    def __init__(self, cutout_prob=0.5, blur_prob=0.5, cutout_size=16, blur_radius=2):
        self.cutout_prob = cutout_prob
        self.blur_prob = blur_prob
        self.cutout_size = cutout_size
        self.blur_radius = blur_radius

    def __call__(self, img):
        if random.random() < self.cutout_prob:
            img = self.cutout(img)

        if random.random() < self.blur_prob:
            img = self.blur(img)

        return img

    def cutout(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.uint8)

        y = random.randint(0, h - self.cutout_size)
        x = random.randint(0, w - self.cutout_size)

        mask[y:y+self.cutout_size, x:x+self.cutout_size] = 0

        img = img * mask.unsqueeze(0).unsqueeze(0)
        return img

    def blur(self, img):
        img_np = img.numpy()
        img_np = cv2.GaussianBlur(img_np.transpose(1, 2, 0), (self.blur_radius * 2 + 1, self.blur_radius * 2 + 1), 0)
        img = torch.from_numpy(img_np.transpose(2, 0, 1))
        return img

class Blend:
    def __init__(self, blend_prob=0.5, alpha_range=(0.2, 0.8)):
        self.blend_prob = blend_prob
        self.alpha_range = alpha_range

    def __call__(self, image1, image2):
        if torch.rand(1) < self.blend_prob:
            alpha = torch.FloatTensor(1).uniform_(*self.alpha_range)
            blended_image = alpha * image1 + (1 - alpha) * image2
            return blended_image
        else:
            return image1

class RGBPermute:
    def __init__(self, permute_prob=0.5):
        self.permute_prob = permute_prob

    def __call__(self, image):
        if torch.rand(1) < self.permute_prob:
            # 随机排列 RGB 通道
            permuted_image = image[:, torch.randperm(3), :, :]
            return permuted_image
        else:
            return image

class Mixup:
    def __init__(self, mixup_alpha=1.0):
        self.mixup_alpha = mixup_alpha

    def __call__(self, image, label):
        # 随机选择另一张图像和标签
        index = torch.randperm(image.size(0))
        mixed_image = self.mixup_alpha * image + (1 - self.mixup_alpha) * image[index]
        mixed_label = self.mixup_alpha * label + (1 - self.mixup_alpha) * label[index]

        return mixed_image, mixed_label

class CutMix:
    def __init__(self, cutmix_alpha=1.0):
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, image, label):
        # 随机选择另一张图像和标签
        index = torch.randperm(image.size(0))
        mixed_image = image.clone()
        mixed_label = label.clone()

        # 随机选择剪切的区域
        lam = torch.rand(1)
        lam = max(lam, 1 - lam)
        cut_ratio = int(image.size(2) * lam.item())

        # 随机选择剪切的位置
        cx = torch.randint(0, image.size(2) - cut_ratio, (1,))
        cy = torch.randint(0, image.size(3) - cut_ratio, (1,))

        # 剪切并贴在另一张图像上
        mixed_image[:, :, cx:cx+cut_ratio, cy:cy+cut_ratio] = image[index, :, cx:cx+cut_ratio, cy:cy+cut_ratio]
        mixed_label[:, :, cx:cx+cut_ratio, cy:cy+cut_ratio] = label[index, :, cx:cx+cut_ratio, cy:cy+cut_ratio]

        return mixed_image, mixed_label