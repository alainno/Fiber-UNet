from statistics import mean
from types import new_class
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
from utils.dice_score import dice_loss
from utils.dice_score import multiclass_dice_coeff

from sklearn.metrics import confusion_matrix
from eval import compute_IoU

def train_net(net, target='MODEL.pth', epochs=100):
        """ funcion de entrenamiento """
        
        # criterion = nn.SmoothL1Loss()
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

        # hiperparametros:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.96)

        # loss history:
        train_losses = []
        val_losses = []

        # early stopping vars:
        best_prec1 = np.Inf #1e6
        epochs_no_improve = 0
        n_epochs_stop = 10

        best_prec2 = 0

        # train loop:
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0

            with tqdm(total=n_train, desc=f'Train Epoch {epoch+1}/{epochs}') as pbar:
                for batch in train_loader:
                    imgs, true_masks = batch['image'], batch['mask']

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    # true_masks = true_masks.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    masks_pred = net(imgs)
                    
                    loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.n_classes).permute(0,3,1,2).int().float(),
                                        multiclass=True)

                    # masks_pred = torch.softmax(masks_pred, dim=1)
                    # masks_pred = torch.argmax(masks_pred, dim=1)
                    # loss = criterion(masks_pred, true_masks)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    epoch_loss += loss.item() * imgs.size(0)
                    # epoch_loss += loss.item()

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    
            train_losses.append(epoch_loss/len(train))
            # train_losses.append(epoch_loss/len(train_loader))
            
            #validacion:
            net.eval()
            epoch_loss = 0
            dice_score = 0

            # n_classes = 3
            # labels = np.arange(n_classes)
            # cm = np.zeros((n_classes,n_classes))
            
            with tqdm(total=n_val, desc=f'Val Epoch {epoch+1}/{epochs}') as pbar:
                for batch in val_loader:
                    imgs, true_masks = batch['image'], batch['mask']
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    # true_masks = true_masks.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    true_masks = F.one_hot(true_masks, net.n_classes).permute(0,3,1,2).int().float()

                    with torch.no_grad():
                        masks_pred = net(imgs)
                        # loss = criterion(masks_pred, true_masks)

                        masks_pred = F.softmax(masks_pred, dim=1)
                        masks_pred = torch.argmax(masks_pred, dim=1)
                        masks_pred = F.one_hot(masks_pred, net.n_classes).permute(0,3,1,2).int().float()
                        dice_score += multiclass_dice_coeff(masks_pred[:,1:,...],true_masks[:,1:,...], reduce_batch_first=False)

                        # masks_pred = torch.softmax(masks_pred, dim=1)
                        # masks_pred = torch.argmax(masks_pred, dim=1)
                        # for j in range(len(true_masks)):
                        #     true = true_masks[j].cpu().detach().numpy().flatten()
                        #     pred = masks_pred[j].cpu().detach().numpy().flatten()
                        #     cm += confusion_matrix(true, pred, labels=labels)

                    # epoch_loss += loss.item() * imgs.size(0)
                    # epoch_loss += dice_score.item() * imgs.size(0)
                    
                    # pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.set_postfix(**{'prec (batch)': dice_score.item()})
                    pbar.update(imgs.shape[0])

            # val_losses.append(epoch_loss/len(val))
            # val_losses.append(epoch_loss/len(val_loader))
            
            if (epoch+1) % 10 == 0:
                scheduler.step()

            # se guarda el modelo si es mejor que el anterior:
            # prec1 = epoch_loss/n_val
            # is_best = prec1 < best_prec1
            # best_prec1 = min(prec1, best_prec1)

            # class_iou,mean_iou = compute_IoU(cm)
            prec2 = dice_score.item() / len(val_loader)
            is_best = prec2 > best_prec2
            
            if is_best:
                epochs_no_improve = 0
                torch.save(net.state_dict(), target)
                best_prec2 = prec2
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    break 

        print(f'The best Loss (train): {min(train_losses)}')                
        # print(f'The best Loss (val): {min(val_losses)}')
        print(f'The best Prec (val): {best_prec2}')
        
        
        plt.plot(train_losses, label='train loss')
        # plt.plot(val_losses, label='validation loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        #plt.show()
        plt.savefig('loss.png')
        
        return min(train_losses),best_prec2


if __name__ == "__main__":

    # cargando el dataset de imágenes
    dir_img = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/images_filtered/"
    # dir_mask = "/data/aalejo/data_wormbodies/BBBC010_v1_foreground/"
    dir_mask = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/foreground_overlapping/"

    # img_transforms = transforms.Compose([
    img_transforms = transforms.RandomChoice([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5],std=[0.5])
        # transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
    ])

    dataset = BasicDataset(dir_img, dir_mask, scale=1, mask_suffix='_ground_truth', transforms=img_transforms)

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
    net.to(device=device)

    lr = 10**-3
    # lr = 5*(10**-4)
    weight_decay = 5*(10**-6)
    # weight_decay = 0

    error_train,prec = train_net(net, 'checkpoints/MODEL_unet3.pth', epochs = 500)
    # error_train_list.append(error_train)
    # error_list.append(error)
    print('prec:', prec)

    # pruebas
    # test_images = enumerate(test_loader)
    # q = next(test_images)

    # with torch.no_grad():
    #     output = net(q[1]['image'].to(device))

    # predictions = torch.nn.functional.softmax(output, dim=1)
    # pred_labels = torch.argmax(predictions, dim=1) 
    # pred_labels = pred_labels.float()

    # for i in range(4):
    #     plt.subplot(2,4,i+1)
    #     plt.imshow(q[1]['mask'][i], cmap='gray')
    #     plt.subplot(2,4,i+5)
    #     plt.imshow(pred_labels.cpu()[i], cmap='gray')

    # plt.savefig('test.png')

    test_dice = 0
    net.eval()

    for batch in test_loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        true_masks = F.one_hot(true_masks, net.n_classes).permute(0,3,1,2).int().float()

        with torch.no_grad():
            masks_pred = net(imgs)

            masks_pred = F.softmax(masks_pred, dim=1)
            masks_pred = torch.argmax(masks_pred, dim=1)
            masks_pred = F.one_hot(masks_pred, net.n_classes).permute(0,3,1,2).int().float()
            test_dice += multiclass_dice_coeff(masks_pred[:,1:,...],true_masks[:,1:,...], reduce_batch_first=False)

    print('Test Dice:', test_dice.item()/len(test_loader))