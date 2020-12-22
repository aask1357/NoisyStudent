import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import os
import argparse
import json

from utils import get_lr, LR_Lambda, Loader, update_params
from data import load_dataset, NS_DataLoader
from logger import Logger, Printer
from models import load_model, save_model


def cross_entropy(output, target):
    return torch.mean(torch.sum(-target * torch.nn.functional.log_softmax(output, dim=1), dim=1))


def train(model, optimizer, lr_scheduler, dataloader, device, logger, epoch):
    start_time = time.time()
    model.train()  # Set model to training mode
    criterion = cross_entropy
        
    epoch_loss, epoch_acc, total_data_size = 0.0, 0, 0

    # Iterate over data
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)

        epoch_loss += loss.item() * inputs.size(0)
        epoch_acc += torch.sum(preds == labels).item()
        total_data_size += inputs.size(0)

    lr_scheduler.step()                

    epoch_loss /= total_data_size
    epoch_acc /= total_data_size
    logger.log("train", epoch, epoch_loss, epoch_acc)
    logger.log_lr(epoch, get_lr(optimizer))

    end_time = time.time()
    print(f'Train Loss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}\tTime: {end_time - start_time:.2f}sec')


def test(model, dataloader, device, logger, epoch):
    start_time = time.time()
    model.eval()   # Set model to evaluate mode
    criterion = nn.CrossEntropyLoss()
    epoch_loss, epoch_acc, total_data_size = 0.0, 0, 0

    # Iterate over data
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        epoch_loss += loss.item() * inputs.size(0)
        epoch_acc += torch.sum(preds == labels).item()
        total_data_size += inputs.size(0)              

    epoch_loss /= total_data_size
    epoch_acc /= total_data_size
    logger.log("test", epoch, epoch_loss, epoch_acc)
    
    end_time = time.time()
    print(f'\t\tTest Loss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}\tTime: {end_time - start_time:.2f}sec', end='\t')
    
    return epoch_acc


def train_model(model, optimizer, lr_scheduler, dataloaders, start_epoch, end_epoch,
                best_acc, best_epoch, best_model, device, logger, phase, epochs_per_save, save_path,
                augment_epoch, unaugment_epoch):
    start_time_total = time.time()
    saved_epoch = -1
    
    for epoch in range(start_epoch+1, end_epoch+1):
        print(f'Epoch {epoch}/{end_epoch}:', end='\t')

        # Each epoch has a training and validation phase
        train(model, optimizer, lr_scheduler, dataloaders["train"], device, logger, epoch)
        epoch_acc = test(model, dataloaders["test"], device, logger, epoch)
        if epoch_acc > best_acc:
            print("Best so far")
            best_acc = epoch_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
        else:
            print(f'Best epoch: {best_epoch} : {best_acc:4f}')
            
        ## fine tuning with unaugmented dataset ##
        if phase > 0 and epoch % augment_epoch == 0:
            start_time = time.time()
            print("Fine Tuning with an unaugmented dataset")
            dataloader = dataloaders['train']
            dataloader.set_resolution("unaugmented")

            model.train()
            
            epoch_loss, epoch_acc, total_data_size = 0.0, 0, 0
            iter_end, iteration = len(dataloader) * unaugment_epoch, 0
            while iteration < iter_end:
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    _, preds = torch.max(outputs, 1)
                    _, labels = torch.max(labels, 1)

                    iteration += 1
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_acc += torch.sum(preds == labels).item()
                    total_data_size += inputs.size(0)

                    if iteration >= iter_end:
                        break
            epoch_loss /= total_data_size
            epoch_acc /= total_data_size
            logger.log("train", epoch+0.5, epoch_loss, epoch_acc)
            end_time = time.time()
            print(f'\t\tTrain Loss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}\tTime: {end_time - start_time:.2f}sec')
            dataloader.set_resolution("augmented")

            # test fine-tuned model
            epoch_acc = test(model, dataloaders['test'], device, logger, epoch+0.5)
            if epoch_acc > best_acc:
                print("Best so far")
                best_acc = epoch_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())
            else:
                print(f'Best epoch: {best_epoch} : {best_acc:4f}')
            
        if epoch % epochs_per_save == 0:
            save_model(model, optimizer, lr_scheduler, epoch, phase, best_acc, best_epoch, best_model, save_path)
            saved_epoch = epoch

            
    if epoch > saved_epoch:
        save_model(model, optimizer, lr_scheduler, epoch, phase, best_acc, best_epoch, best_model, save_path)
        
    sec = time.time() - start_time_total
    print(f'====Total time: {int(sec)//3600}h {(int(sec)%3600)//60}m {sec%60:.2f}s ({sec:.2f} s)====')
    
    # load best model weights
    model.load_state_dict(best_model)
    return model


def main(output_dir, model_name, batch_size, num_workers, augment_epoch, unaugment_epoch,
          device, label_type, confidence, num_classes, epochs_per_save, teacher_noise,
          data_config, model_config):
    assert label_type in ["soft", "hard"]
    # setup for directory and log
    output_dir = os.path.join(output_dir, model_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o775)
    
    if hasattr(print, "set_log_dir"):
        print.set_log_dir(output_dir)
        
    print(f"==========Settings==========")
    print(f"batch size:      {batch_size}")
    print(f"augment epoch:   {augment_epoch}")
    print(f"unaugment epoch: {unaugment_epoch}")
    print(f"label type:      {label_type}")
    print(f"confidence:      {confidence}")
    print(f"models:          {model_config['models']}")
    print(f"epochs:          {model_config['epochs']}")
    print(f"ratio:           {model_config['ratio']}")
    print(f"learning rate:   {model_config['learning_rate']}")
    print(f"lr_decay_rate:   {model_config['lr_decay_rate']}")
    print(f"lr_decay_epoch:  {model_config['lr_decay_epoch']}")
    print(f"N, M:            {data_config['N']}, {data_config['M']}")
    print(f"teacher noise:   {teacher_noise}")
    print(f"dropout_prob:    {model_config['dropout_prob']}")
    print(f"stoc_depth_prob: {model_config['stochastic_depth_prob']}")
    print("============================")
    
    logger = Logger(os.path.join(output_dir, 'logs'))
    
    # dataset, dataloader
    
    dataset = load_dataset(**data_config)
    dataloaders = {}
    dataloaders['train_unaugmented'] = DataLoader(dataset['train_unaugmented'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['train_augmented'] = DataLoader(dataset['train_augmented'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['test'] = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # load model
    loader = Loader(**model_config, device = device, num_classes = num_classes)
    (teacher, student, optimizer, lr_scheduler, start_epoch, end_epoch, phase,
              best_acc, best_epoch, best_model) = loader.load(0)
    model_name = model_config["models"][phase]
    ratio = model_config["ratio"][phase]
    save_path = os.path.join(output_dir, f"{model_name}.pt")
    logger.set_model_name(model_name)
    
    ############################
    ###### Training phase ######
    ############################
    if teacher is None:
        dataloaders['train'] = dataloaders['train_augmented'] if teacher_noise else dataloaders['train_unaugmented']
    else:
        dataloaders['train'] = NS_DataLoader(dataloaders, dataset, teacher, device, label_type, confidence, ratio, num_workers, batch_size, num_classes, print)
    
    while True:
        teacher = train_model(student, optimizer, lr_scheduler, dataloaders, start_epoch, end_epoch, 
                            best_acc, best_epoch, best_model, device, logger, phase, epochs_per_save, save_path,
                            augment_epoch, unaugment_epoch)
        phase += 1
        if phase < len(model_config["models"]):
            (_, student, optimizer, lr_scheduler, start_epoch, end_epoch, _,
                best_acc, best_epoch, best_model) = loader.load(phase)
            model_name = model_config["models"][phase]
            ratio = model_config["ratio"][phase]
            
            logger.set_model_name(model_name)
            save_path = os.path.join(output_dir, f"{model_name}.pt")
            
            dataloaders['train'] = NS_DataLoader(dataloaders, dataset, teacher, device, label_type, confidence, ratio, num_workers, batch_size, num_classes, print)
            del teacher
        else:
            break
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()
    args.rank = 0

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, args.params)
    
    train_config = config["train_config"]
    model_config = config["model_config"]
    data_config = config["data_config"]
    
    print = Printer()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(**train_config, model_config = model_config, data_config = data_config)
