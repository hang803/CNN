import random
import numpy as np
import torch
import math


def mixup(x, y, alpha):
    batch = len(y)
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch)
    mixup_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixup_x, y_a, y_b, lam


def cutmix(X, Y, alpha):
    lam = np.random.beta(alpha, alpha)
    H, W = X.size()[-2:]
    x = np.random.randint(H)
    y = np.random.randint(W)
    w = W * (1 - lam) ** 0.5
    h = H * (1 - lam) ** 0.5
    x1 = int(np.clip(x - w // 2, 0, W))
    x2 = int(np.clip(x + w // 2, 0, W))
    y1 = int(np.clip(y - h // 2, 0, H))
    y2 = int(np.clip(y + h // 2, 0, H))
    index = torch.randperm(X.size()[0])
    X[:, :, x1:x2, y1:y2] = X[index, :, x1:x2, y1:y2]
    lam1 = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    y_a, y_b = Y, Y[index]
    return X, y_a, y_b, lam1


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


class RandomErasing:
    def __init__(self, p=0.5, sl=0.02, sh=0.1, r1=0.3):
        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1 / r1)

    def __call__(self, img):
        C,H,W=img.shape
        if random.random() > self.p:
            return img
        else:
            Se = random.uniform(*self.s) * W*H
            re = random.uniform(*self.r)

            He = int(round(math.sqrt(Se * re)))
            We = int(round(math.sqrt(Se / re)))

            xe = random.randint(0, img.size()[-1])
            ye = random.randint(0, img.size()[-2])

            if xe + We <= W and ye + He <= H:
                img[:,ye: ye + He, xe: xe + We]=torch.randint(0,1,[C,He,We])
                # if C==3:
                #     img[0,ye: ye + He, xe: xe + We]=1
                #     img[1, ye: ye + He, xe: xe + We] = 1
                #     img[2, ye: ye + He, xe: xe + We] = 1
                # else:
                #     img[0, ye: ye + He, xe: xe + We] = 1
            return img
