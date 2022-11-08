from unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # cargando el dataset de imágenes
    dir_img = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/images_filtered/"
    # dir_mask = "/data/aalejo/data_wormbodies/BBBC010_v1_foreground/"
    dir_mask = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/foreground_overlapping/"

    dataset = BasicDataset(dir_img, dir_mask, scale=1, mask_suffix='_ground_truth')

    val_percent = 0.2
    batch_size = 4

    n_val = int(len(dataset) * val_percent)
    n_test = 5
    n_train = len(dataset) - n_val - n_test

    train, val, test = random_split(dataset, [n_train, n_val, n_test])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using ", device)

    net = UNet(n_channels=1, n_classes=3, bilinear=False)
    net.load_state_dict(torch.load("checkpoints/MODEL_unet3.pth"))
    net.to(device=device)
    net.eval()

    # pruebas
    test_images = enumerate(test_loader)
    q = next(test_images)
    # show gt
    # plt.figure(figsize=(15,15))
    # for i in range(4):
    #     plt.subplot(1, 4, i+1)
    #     plt.imshow(q[1]['mask'][i], cmap='gray')
    # plt.savefig('test-gt.png')

    with torch.no_grad():
        output = net(q[1]['image'].to(device))

    predictions = torch.nn.functional.softmax(output, dim=1)
    pred_labels = torch.argmax(predictions, dim=1) 
    pred_labels = pred_labels.float()

    for i in range(4):
        plt.subplot(2,4,i+1)
        plt.imshow(q[1]['mask'][i], cmap='gray')
        plt.subplot(2,4,i+5)
        plt.imshow(pred_labels.cpu()[i], cmap='gray')

    plt.savefig('outputs/test.png')

    test_dice = 0
    print("Test Dice:", test_dice)