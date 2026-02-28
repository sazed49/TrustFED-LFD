import torch
from torch.utils import data
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class = None, target_class = None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class  
        self.contains_source_class = False
            
    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class = None, target_class = None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class  
            
    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.dataset)

    
class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array
        
        Return xtrain and ylabel in torch tensor datatype
        """
        self.reviews = reviews
        self.target = targets
    
    def __len__(self):
        # return length of dataset
        return len(self.reviews)
    
    def __getitem__(self, index):
        # given an index (item), return review and target of that index in torch tensor
        x = torch.tensor(self.reviews[index,:], dtype = torch.long)
        y = torch.tensor(self.target[index], dtype = torch.float)
        
        return  x, y

# A method for combining datasets  
def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)



class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Path'])  # Make sure 'Path' column is relative
        label = int(row['ClassId'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

def get_gtsrb():
    train_csv = 'data/gtsrb/Train.csv'
    test_csv = 'data/gtsrb/Test.csv'
    root_dir = 'data/gtsrb'  # This is the base path for all images

    trainset = GTSRBDataset(train_csv, root_dir)
    testset = GTSRBDataset(test_csv, root_dir)
    return trainset, testset
class MineSignsDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = img_path.replace(".jpg", ".txt")

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        with open(label_path) as f:
            line = f.readline().strip()
            cls, xc, yc, w, h = map(float, line.split())

        # Convert YOLO → pixel bbox
        x1 = int((xc - w/2) * W)
        y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W)
        y2 = int((yc + h/2) * H)

        cropped = image.crop((x1, y1, x2, y2))

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, int(cls)
class MineSignDataset2:

        def __init__(self, root_dir, batch_size=32):

            self.root_dir = root_dir
            self.batch_size = batch_size

            self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        def get_train_loader(self):

            train_path = os.path.join(self.root_dir, "train")

            train_dataset = ImageFolder(
            root=train_path,
            transform=self.transform
        )

            return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        def get_val_loader(self):

            val_path = os.path.join(self.root_dir, "val")

            val_dataset = ImageFolder(
            root=val_path,
            transform=self.transform
        )

            return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
def get_minesigns():
    root = "S:/Summer25/MineDataset/Annotation_Done/MNIST_Format"  # has train/ and val/

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = ImageFolder(root + "/train", transform=transform)
    testset  = ImageFolder(root + "/val",   transform=transform)

    return trainset, testset

