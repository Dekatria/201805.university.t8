from PIL import Image
from torch import Tensor, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import *
from getimagenetclasses import parsesynsetwords, parseclasslabel
import sys


class cropSet(Dataset):

    def __init__(self, path, xmlDir, size):
        self.path = path
        self.xmlDir = xmlDir
        self.size = size
        self.transform = None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        path = self.path.format(idx + 1)
        im = Image.open(path).convert("RGB")

        idx, name = parseclasslabel(self.xmlDir.format(idx + 1), syn2idx)

        if self.transform:
            im = self.transform(im)

        return im, idx


def center(img):
    C, W, H = img.size()
    crop = img.new_empty((3, 224, 224))
    x = img.narrow(1, (W - 224) // 2, 224)
    crop = x.narrow(2, (H - 224) // 2, 224)

    return crop


class centerSet(cropSet):

    def __init__(self, *arg):
        super().__init__(*arg)
        self.transform = \
            Compose([
                Resize(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
                center,
            ])


def five(img):
    C, W, H = img.size()
    crop5 = img.new_empty((5, 3, 224, 224))
    x = img.narrow(1, 0, 224)
    x = x.narrow(2, 0, 224)
    crop5[0] = x
    x = img.narrow(1, W - 224, 224)
    x = x.narrow(2, 0, 224)
    crop5[1] = x
    x = img.narrow(1, W - 224, 224)
    x = x.narrow(2, H - 224, 224)
    crop5[2] = x
    x = img.narrow(1, 0, 224)
    x = x.narrow(2, H - 224, 224)
    crop5[3] = x
    x = img.narrow(1, (W - 224) // 2, 224)
    x = x.narrow(2, (H - 224) // 2, 224)
    crop5[4] = x

    return crop5


class fiveSet(cropSet):

    def __init__(self, *arg):
        super().__init__(*arg)
        self.transform = \
            Compose([
                Resize(280),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
                five,
            ])


def test(model, dataloader):
    total = 0
    correct = 0
    for n, sample_batched in enumerate(dataloader):
        data, descs = sample_batched
        X = data.view(-1, 3, 224, 224)
        out = model.forward(X)
        out = out.view(batch_size, -1, 1000)
        out = out.mean(1)
        val, pred = out.max(1)
        cmp = pred.eq(descs)
        total += cmp.size(0)
        correct += cmp.sum()

    return int(correct), total

def main(dataDir, xmlDir, synDir, n=250, batch_size=10, is_shuffle=False):
    idx2syn, syn2idx, syn2desc = parsesynsetwords(synDir)
    fivecropset = fiveSet(dataDir, xmlDir, n)
    fivecroploader = DataLoader(
        fivecropset, batch_size=batch_size, shuffle=is_shuffle)
    centercropset = centerSet(dataDir, xmlDir, n)
    centercroploader = DataLoader(
        centercropset, batch_size=batch_size, shuffle=is_shuffle)
    model = resnet18(pretrained=True)
    model.eval()
    fiveResult = test(model, fivecroploader)
    centerResult = test(model, centercroploader)

    return fiveResult, centerResult


if __name__ == "__main__":
    fiveResult, centerResult = main(sys.argv[1], sys.argv[2], sys.argv[3])
    print("Five Crop Accuracy: %s" % fiveResult)
    print("Center Crop Accuracy: %s" % centerResult)

    
