'''
00 - The Fool 
Fooling classifier code
for homework 5
'''
import torch
import PIL.Image as Image
import PIL.ImageChops as ImChop

import torchvision
from torchvision import models, transforms, utils

import numpy as np
import matplotlib.pyplot as plt
import getimagenetclasses as ginc

model_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

inv_transform = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                         std=[1., 1., 1.]),
                                    transforms.ToPILImage(),
                                    ])

resize_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)
])

i2s, s2i, s2d = ginc.parsesynsetwords('synset_words.txt')


def get_descript(class_no):
    s = i2s[class_no]
    desc = s2d[s]
    return desc


def get_trained_resnet(use_gpu=True):
    model_ft = models.resnet18(pretrained=True)
    if use_gpu:
        model_ft = model_ft.try_cuda()
    return model_ft


def plot_with_difference(original, fooled):
    '''
    Takes the normalised tensor of two images and plot them
    '''
    fooled_v = fooled.clone().view(3, 224, 224).cpu().detach()
    fooled_back = inv_transform(fooled_v)
    original_back = inv_transform(original.clone())

    plt.figure()
    plt.subplot(131)
    plt.imshow(original_back)
    plt.title('original')

    boi_diff = ImChop.difference(fooled_back, original_back)
    plt.subplot(132)
    plt.imshow(boi_diff)
    plt.title('difference')

    plt.subplot(133)
    plt.imshow(fooled_back)
    plt.title('fooling')
    print("Total value difference:", np.array(boi_diff).sum(),
          "\nAverage value difference:", np.array(boi_diff).mean())


# add function to torch.Tensor class
# python gets stuck after getting the second cuda error
# so after the first error, this function should always
# return self
if __name__ == "__main__":
    have_cuda = True

    def try_cuda(self):
        global have_cuda
        if have_cuda:
            try:
                with_cuda = self.cuda()
                return with_cuda
            except Exception as e:
                have_cuda = False
                print("try_cuda failed:", e, "\n proceeding without cuda")
        return self

    torch.Tensor.try_cuda = try_cuda
    torch.nn.Module.try_cuda = try_cuda
