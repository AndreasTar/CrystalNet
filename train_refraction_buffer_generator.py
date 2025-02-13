import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
from sklearn.model_selection import train_test_split
from ext import pytorch_ssim
from utils import DL1Combine, save_checkpoint, load_checkpoint
from net_models import CrystalNet as RBufferGenerator
import torch.optim.lr_scheduler

def load_data(dataset_name, obj_num, batch_size, res=256, test_size=0.03, random_seed=42):
    """Loads and splits the dataset into train and validation sets."""
    # Load data

    logging.info("Loading Datasets...")

    try:
        G = torch.load(f"./datasets/{dataset_name}/{dataset_name}_RAOV_Xg_{res}.pt", mmap=True)
        X = torch.load(f"./datasets/{dataset_name}/{dataset_name}_RAOV_X_{res}.pt", mmap=True)
        Rnpz = np.load(f"./datasets/{dataset_name}/{dataset_name}_RAOV_{res}.npz", mmap_mode='r')
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise

    X = torch.moveaxis(X,3,1)
    
    logging.info("Finished Loading Datasets.")
    logging.info("Altering shape of data and matrices...")

    Rnp = np.rollaxis(Rnpz["RAOV"], 3, 1)
    Rnp[:,0,...] = np.round(Rnp[:,0,...])
    Rnp[:,0,...] += 1
    mask = np.expand_dims(np.clip(np.round(torch.sum(torch.abs(G[:,:,8,:,:]),dim=1).numpy()),0,1),axis=1)
    Rnp = (Rnp*mask)
    Rnp[:,0,...][Rnp[:,0,...] > obj_num] = 0
    Rnp[:,0,...][Rnp[:,0,...] < 0] = 0

    data_length = X.shape[0]
    train_length = int(data_length*test_size)

    logging.info("Beggining Train/Validation Splits...")
        
    # Because of the large volume of data, the train/validation split will be seperated by a fixed range
    # This won't affect anything because the data itself is iid (independent & identically distributed), 
    # although it will be painful to switch validation set. More elegant approach will come later.

    Xtr = X[:train_length,...]
    Xva = X[train_length:,...]

    Gtr = G[:train_length,...]
    Gva = G[train_length:,...]

    Rtr = torch.Tensor(Rnp[:train_length,...])
    Rva = torch.Tensor(Rnp[train_length:,...])

    logging.info(f"Shape of train data : {Xtr.shape}")

    exit(999)

    logging.info("Coverting to Tensors and Creating DataLoaders...")
    
    # Convert to tensors
    dataset_tr = TensorDataset(Xtr,Gtr,Rtr[:,-3:,...],Rtr[:,0,...],Rtr[:,1:3,...])
    dataset_va = TensorDataset(Xva,Gva,Rva[:,-3:,...],Rva[:,0,...],Rva[:,1:3,...])
    
    # DataLoader creation
    dataloader_train = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)

    logging.info("Finished Loading Data!")
    
    return dataloader_train, dataloader_val

def train_model(net, dataloader_train, dataloader_val, optimizer, scheduler, criterion, num_epochs, device, ds_name):
    """Main training loop."""
    net.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x_raw, g, sn,oi,uv in dataloader_train:
            x, g = x_raw.to(device), g.to(device)
            oi = oi.to(device).long()
            uv = uv.to(device)
            sn = sn.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred_sn, pred_oi, pred_uv = net(x, g)
            loss = criterion(pred_sn, pred_oi, pred_uv, sn, oi, uv)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader_train)
        if epoch % 2 == 0:
            logging.info(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Validation and Checkpointing
        if epoch % 10 == 0:
            val_loss = evaluate_model(net, dataloader_val, criterion, device)
            logging.info(f"Validation Loss: {val_loss:.4f}")

        # Save model checkpoints
        if epoch % 20 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir="./models", filename=f"{ds_name}_checkpoint_epoch_{epoch}.pth")

        # Step the scheduler
        scheduler.step()

def evaluate_model(net, dataloader_val, criterion, device):
    """Evaluates the model on the validation set."""
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for x_raw, g_val, sn_val, oi_val, uv_val in dataloader_val:
            # Move data to the device
            x_val, g_val = x_raw.to(device), g_val.to(device)
            oi_val = oi_val.to(device).long()
            uv_val = uv_val.to(device)
            sn_val = sn_val.to(device)
            
            # Forward pass
            pred_sn, pred_oi, pred_uv = net(x_val, g_val)
            
            # Calculate loss
            loss = criterion(pred_sn, pred_oi, pred_uv, sn_val, oi_val, uv_val)
            val_loss += loss.item()
    
    # Return average validation loss
    return val_loss / len(dataloader_val)

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-name', type=str, required=True)
    parser.add_argument('--num-idx', type=int, required=True, help='Number of objects in the scene')
    parser.add_argument('--single-batch-size', type=int, default=3) # typical sizes 2 - 64
    parser.add_argument('--num-epochs', type=int, default=100)      # typical epochs 50 - ?
    args = parser.parse_args()
    
    # Logging Configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')
    
    # Load Data
    batch_size = torch.cuda.device_count() * args.single_batch_size
    dataloader_train, dataloader_val = load_data(args.scene_name, args.num_idx, batch_size)

    # Model, Optimizer, Scheduler, and Criterion
    net = RBufferGenerator(args.num_idx+1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1.0e-4, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.9)
    criterion = DL1Combine()
    
    # Training Loop
    train_model(net, dataloader_train, dataloader_val, optimizer, scheduler, criterion, args.num_epochs, device, args.scene_name)

if __name__ == "__main__":
    main()
