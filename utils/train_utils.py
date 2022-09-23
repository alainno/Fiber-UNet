import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train_net(net, device, epochs, learning_rate, weight_decay, target='MODEL.pth'):
    """ funcion de entrenamiento """
    
    # criterion = nn.SmoothL1Loss()
    criterion = nn.BCEWithLogitsLoss()

    # hiperparametros:
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.96)

    # loss history:
    train_losses = []
    val_losses = []

    # early stopping vars:
    best_prec1 = np.Inf #1e6
    epochs_no_improve = 0
    n_epochs_stop = 10

    # train loop:
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Train Epoch {epoch+1}/{epochs}') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch['image'], batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                
                loss = criterion(masks_pred, true_masks)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                epoch_loss += loss.item() * imgs.size(0)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
        train_losses.append(epoch_loss/len(train))
        
        #validacion:
        net.eval()
        epoch_loss = 0
        
        with tqdm(total=n_val, desc=f'Val Epoch {epoch+1}/{epochs}') as pbar:
            for batch in val_loader:
                imgs, true_masks = batch['image'], batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    mask_pred = net(imgs)
                    loss = criterion(mask_pred, true_masks)
                
                epoch_loss += loss.item() * imgs.size(0)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])

        val_losses.append(epoch_loss/len(val))
        
        if (epoch+1) % 10 == 0:
            scheduler.step()

        # se guarda el modelo si es mejor que el anterior:
        prec1 = epoch_loss/n_val
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        
        if is_best:
            epochs_no_improve = 0
            torch.save(net.state_dict(), target) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                break    

    print(f'The best Loss (train): {min(train_losses)}')                
    print(f'The best Loss (val): {min(val_losses)}')
    
    
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    return min(train_losses),best_prec1