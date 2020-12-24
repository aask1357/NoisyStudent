import torch
import torchvision
from torchvision.datasets import STL10
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import os
from tqdm import tqdm
import time

from RandAugment import RandAugment


def one_hot(label):
    oh = torch.zeros(10)
    oh[label] = 1.0
    return  oh

    
def load_dataset(N, M, dataset, dataset_dir):
    '''Add RandAugment with N, M(hyperparameter)
    N: Number of augmentation transformations to apply sequentially.
    M: Magnitude for all the transformations.'''
    transform_randaugment = transforms.Compose([
        RandAugment(N, M),
        #transforms.Resize(32),
        transforms.ToTensor(),
    ])
    transform_resize = transforms.Compose([
        #transforms.Resize(32),
        transforms.ToTensor(),
    ])
    
    if dataset == 'stl10':
        stl10 = {}
        stl10['train_augmented'] = STL10(dataset_dir, 'train', transform=transform_randaugment,
                                         target_transform=one_hot, download=True)
        stl10['train_unaugmented'] = STL10(dataset_dir, 'train', transform=transform_resize,
                                            target_transform=one_hot)
        stl10['unlabeled_augmented'] = STL10(dataset_dir, 'unlabeled', transform=transform_randaugment, download=True)
        stl10['unlabeled_unaugmented'] = STL10(dataset_dir, 'unlabeled', transform=transform_resize, download=True)
        stl10['test'] = STL10(dataset_dir, 'test', transform=transform_resize, download=True)

        return stl10
    else:
        raise Excpetion(f"Dataset '{dataset}' not implemented")


class NS_Dataset(Dataset):
    def __init__(self, dataset, table):
        self.dataset = dataset
        self.table = table
        
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        x, _ = self.dataset[self.table[idx][0]]
        label = self.table[idx][1]
        
        return x, label
        
        
class NS_DataLoader:
    def __init__(self, dataloaders, datasets, model, device, label_type, confidence, ratio, num_workers, batch_size, num_classes, print):
        self.dataloader= {"augmented": {}, "unaugmented": {}}
        self.dataloader["augmented"]["labeled"] = dataloaders["train_augmented"]
        self.dataloader["unaugmented"]["labeled"] = dataloaders["train_unaugmented"]
        self.resolution = "augmented"
        
        self.label_data(dataloaders, datasets, model, device, label_type, confidence, ratio, num_workers, batch_size, num_classes, print)
        
        self.len_labeled = len(self.dataloader["augmented"]["labeled"])
        self.len_unlabeled = len(self.dataloader["augmented"]["unlabeled"])
        self.iter_unlabeled = iter(self.dataloader["augmented"]["unlabeled"])
        self.idx_labeled = 0
        self.idx_unlabeled = 0
    
    def label_data(self, dataloaders, datasets, model, device, label_type, confidence, ratio, num_workers, batch_size, num_classes, print):
        start_time = time.time()
        print("Labeling unlabeled data...")
        assert label_type in ["soft", "hard"]
        
        # label with unaugmented unlabeld dataset
        data_size = len(datasets["unlabeled_unaugmented"])
        dataloader = DataLoader(datasets['unlabeled_unaugmented'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        model.eval()
        count = [0] * num_classes
        table = [[] for _ in range(num_classes)]
        for idx, (x, _) in enumerate(tqdm(dataloader)):
            x = x.to(device)
            with torch.no_grad():
                outputs = model(x)
                outputs =  F.softmax(outputs, dim=1).to('cpu')
                ps, labels = torch.max(outputs, dim=1)
                for i, (p, output, label) in enumerate(zip(ps, outputs, labels)):
                    if p.item() >= float(confidence):
                        if label_type == "soft":
                            table[label].append([idx*batch_size + i, output, p.item()])
                        else:
                            hard_label = one_hot(label)
                            table[label].append([idx*batch_size + i, hard_label, p. item()])
                        count[label] += 1
        
        print(count)
        # make sure that total count per class is less than real count per class
        count_per_class = min(max(count), data_size//num_classes)
        
        print("[", end='')
        for i in range(num_classes):
            table[i] = sorted(table[i], key=lambda t: t[2], reverse=True)
            to_add = count_per_class - count[i]
            while to_add > 0:
                table[i].extend(table[i][0:to_add])
                to_add = count_per_class - len(table[i])
            if to_add < 0:
                table[i] = table[i][:to_add]
            print(f"{len(table[i])}", end=', ' if i < num_classes - 1 else '')
        print("]")
        
        self.table = []
        for t in table:
            for e in t:
                self.table.append(e[0:2])  # drop p
        
        dataset1 = NS_Dataset(datasets["unlabeled_unaugmented"], self.table)
        dataset2 = NS_Dataset(datasets["unlabeled_augmented"], self.table)
        self.dataloader["unaugmented"]["unlabeled"] = DataLoader(dataset1, batch_size=int(batch_size*ratio), shuffle=True, num_workers=num_workers)
        self.dataloader["augmented"]["unlabeled"] = DataLoader(dataset2, batch_size=int(batch_size*ratio), shuffle=True, num_workers=num_workers)
        
        print(f"Labeling completed in {time.time() -start_time:.2f}sec")
    
    def set_resolution(self, resolution):
        assert resolution in ["augmented", "unaugmented"]
        self.resolution = resolution
        self.idx_unlabeled = 0
        self.iter_unlabeled = iter(self.dataloader[resolution]["unlabeled"])
    
    def __len__(self):
        return self.len_labeled
    
    def __iter__(self):
        self.idx_labeled = 0
        self.iter_labeled = iter(self.dataloader[self.resolution]["labeled"])
        return self
    
    def __next__(self):
        if self.idx_labeled >= self.len_labeled:
            raise StopIteration
        
        x1, label_1 = next(self.iter_labeled)
        
        if self.idx_unlabeled >= self.len_unlabeled:
            self.idx_unlabeled = 0
            self.iter_unlabeled = iter(self.dataloader[self.resolution]["unlabeled"])
        x2, label_2 = next(self.iter_unlabeled)
        
        x = torch.cat((x1, x2), dim=0)
        label = torch.cat((label_1, label_2), dim=0)
            
        self.idx_labeled += 1
        self.idx_unlabeled += 1
        
        return x, label
    
    
