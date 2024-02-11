# dataloader for MNIST_Q, CIFAR_Q, TMED, and private AS datasets
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as pl

from typing import Any, Callable, Optional, Tuple

DATASET_ROOT = "/workspace/datasets/"

class CIFAR_Q_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_root=DATASET_ROOT, download=False, img_resolution=224, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.data_root = data_root
        self.download = download
        self.img_resolution = img_resolution
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.Resize(size=(self.img_resolution, self.img_resolution)),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
    
    def setup(self, stage):
        self.ds_test = CIFAR10_Quadrants(self.data_root, train=False, transform=self.transform, download=self.download)
        self.ds_predict = CIFAR10_Quadrants(self.data_root, train=False, transform=self.transform, download=self.download)
        self.ds_train = CIFAR10_Quadrants(self.data_root, train=True, transform=self.transform, download=self.download)
        # TODO form training and validation splits, now we are just using test set as validation
        # Subsets do not inherit any novel properties, such as pseudolabels, or classes, so we have to dance around it
        
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.ds_predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def get_pseudo(self):
        return self.ds_train.get_pseudo()
    
    def set_pseudo(self, pseudo):
        # modify the pseudo property of ds_train
        self.ds_train.set_pseudo(pseudo)
        
    def get_classes(self):
        # return the names of each class
        return self.ds_train.classes
        

def index_to_quadrant_id(index):
    # works w/ integers and numpy arrays
    img_index = index // 4
    quadrant_id = index % 4
    return img_index, quadrant_id

# function to extract quadrant from PIL image
def get_quadrant(img, quadrant_id):
    # PIL image -> PIL image
    w, h = img.size
    match quadrant_id:
    # quadrant ID is 0:top_left, 1:top_right, 2:bottom_left, 3:bottom_right
        case 0:
            window = (0, 0, w//2, h//2)
        case 1:
            window = (w//2, 0, w, h//2)
        case 2:
            window = (0, h//2, w//2, h)
        case 3:
            window = (w//2, h//2, w, h)
        case _:
            print("Invalid quadrant ID")
    img = img.crop(window)
    img = img.resize((w,h))
    return img
    

class CIFAR10_Quadrants(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.init_pseudo()
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Accesses a quadrant of an image in CIFAR
        Args:
            index (int): quadrant-based index, which is equal to 4*image_index + quadrant ID
            quadrant ID is 0:top_left, 1:top_right, 2:bottom_left, 3:bottom_right
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_index, quadrant_id = index_to_quadrant_id(index)
        img = self.data[img_index]
        y = int(self.targets[img_index]) # human-assigned GT
        y_u = self.pseudo[index] # uncertainty-augmented target

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = get_quadrant(img, quadrant_id)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'x':img, 'y':y, 'y_u':y_u, 'uid':str(index)}

    def __len__(self):
        return len(self.data) * 4
        
    def init_pseudo(self):
        len_data = len(self.data) * 4
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # initialize a dictionary to keep track of data
        self.pseudo = {}
        # iterate through the dataset to create a mapping between index and pseudolabel
        for i in range(len_data):
            img_index, _ = index_to_quadrant_id(i)
            label = int(self.targets[img_index])
            self.pseudo[i] = np.zeros(len(self.classes))
            self.pseudo[i][label] = 1.0
        
    def get_pseudo(self):
        return self.pseudo
        
    def set_pseudo(self, new_pseudo):
        for k in new_pseudo.keys():
            if self.pseudo.get(k):
                self.pseudo[k] = new_pseudo[k]
        
if __name__ == "__main__":
    data_config = {"batch_size":4, "data_root":DATASET_ROOT, "download":False, "img_resolution":224}
    dm = CIFAR_Q_DataModule(**data_config)
    stage = "fit"
    dm.setup(stage)
    train_loader = dm.train_dataloader()
    #val_loader = dm.val_dataloader()
    classes = dm.get_classes()
    nc = len(classes)
    
    # try to iterate over the training set, simulate getting new pseudolabels, and printing labels again
    for n_evolution_epoch in range(2):
        pseudolabels = {}
        for i, (img, target, index) in enumerate(train_loader):
            if i < 10:
                print(index)
                print(target)
                for b in range(data_config["batch_size"]):
                    pseudolabel = np.full(nc, 1/nc) * 0.1 + target[b].numpy() * 0.9
                    pseudolabels[index[b].item()] = pseudolabel
            else: # on the second iteration, there should be an error here as we try to index past the pseudolabel range
                break
        print(pseudolabels)
        dm.set_pseudo(pseudolabels)