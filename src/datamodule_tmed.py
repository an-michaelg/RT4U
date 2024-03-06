# dataloader for TMED_2 datasets
import os
import re
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from typing import List, Dict, Union
from typing import Any, Callable, Optional, Tuple

from data_transforms import AdjustGamma

DATASET_ROOT = "/workspace/datasets/"

tmed_label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'tufts':  {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS':1, 'moderate_AS': 2, 'severe_AS': 2},
    'binary': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS':1, 'moderate_AS': 1, 'severe_AS': 1},
    'all': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS':2, 'moderate_AS': 3, 'severe_AS': 4},
    'not_severe': {'no_AS': 0, 'mild_AS': 0, 'mildtomod_AS':0, 'moderate_AS': 0, 'severe_AS': 1}
}

tmed_view_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'A4C': 0, 'A2C': 0, 'PLAX':1, 'PSAX': 1, 'A4CorA2CorOther': 0},
    'all': {'A4C': 0, 'A2C': 1, 'PLAX':2, 'PSAX': 3, 'A4CorA2CorOther': 4}
}

class TMED2_DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_root: str = DATASET_ROOT, 
        batch_size: int = 16,
        parasternal_only: bool = True, 
        label_scheme_name: str = "all",
        img_resolution: int = 224, 
        transform_min_crop_ratio: float = 0.9,
        transform_rotate_degrees: float = 15,
        sampler: str = "random", # random/AS
        num_workers: int = 0
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.parasternal_only = parasternal_only
        self.label_scheme_name = label_scheme_name
        self.img_resolution = img_resolution
        self.num_workers = num_workers
        self.sampler = sampler
        
        # Regarding what Tufts does: they don't go farther than saying "random crops and flips"
        tfd = transforms.Compose([
           transforms.Resize(size=(self.img_resolution, self.img_resolution)),
           AdjustGamma(const_gamma=2.0),
           transforms.ToTensor(), 
           transforms.Normalize((0.1122,0.1122,0.1122),(0.0321,0.0321,0.0321))
           ])
        self.transform_deterministic = tfd
        tfr = transforms.Compose([
           transforms.RandomResizedCrop(size=(self.img_resolution, self.img_resolution), scale=(transform_min_crop_ratio, 1.0)),
           transforms.RandomRotation(degrees=transform_rotate_degrees),
           AdjustGamma(const_gamma=2.0),
           transforms.ToTensor(), 
           transforms.Normalize((0.1122,0.1122,0.1122),(0.0321,0.0321,0.0321))
           ])
        self.transform_random = tfr
    
    def setup(self, stage):
        self.ds_test = self.get_AS_dataset(split="test", transform=self.transform_deterministic)
        self.ds_predict = self.get_AS_dataset(split="test", transform=self.transform_deterministic)
        self.ds_val = self.get_AS_dataset(split="val", transform=self.transform_deterministic)
        self.ds_train = self.get_AS_dataset(split="train", transform=self.transform_random)
            
    def get_AS_dataset(self, split, transform):
        dset = TMED2(self.data_root, 
            split=split, 
            parasternal_only=self.parasternal_only, 
            label_scheme_name=self.label_scheme_name, 
            transform=transform
        )
        print(f"New dataset instantiated: {split}")
        return dset
        
    def train_dataloader(self):
        if self.sampler == "AS":
            sampler_AS = self.ds_train.class_sampler()
            loader = DataLoader(self.ds_train, batch_size=self.batch_size, sampler=sampler_AS, num_workers=self.num_workers)
        else:
            loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def predict_dataloader(self):
        loader = DataLoader(self.ds_predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader
        
    def get_pseudo(self):
        return self.ds_train.get_pseudo()
    
    def set_pseudo(self, pseudo):
        # modify the pseudo property of ds_train
        self.ds_train.set_pseudo(pseudo)
        
    def get_classes(self):
        # return the names of each class
        return self.ds_train.classes
        
def load_image(path):
    im = Image.open(path)
    im = im.convert("RGB")
    return im # (112, 112)
    
    
def studyinfo_from_query_key(s):
    # Define the regular expression pattern
    pattern = r'(\d+)s(\d+)_(\d+)\.png'
    
    # Use re.match to search for the pattern in the string
    match = re.match(pattern, s)
    
    # If a match is found, extract the groups and return them as a tuple
    if match:
        ID, studyNum, imageNum = match.groups()
        return int(ID), int(studyNum), int(imageNum)
    else:
        return None
        

class TMED2(Dataset):
    def __init__(
        self,
        dataset_root: str = DATASET_ROOT,
        split: str = "train", # train/val/test
        parasternal_only: bool = True, 
        label_scheme_name: str = "all",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        
        # obtain dataframe
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        df_path = os.path.join(dataset_root, "DEV479", "TMED2_fold0_labeledpart.csv")
        df = pd.read_csv(df_path)
        
        # filter out the samples without diagnosis label
        df = df[df['diagnosis_label']!='Not_Provided']
        
        # keep only parasternal views
        if parasternal_only:
            df = df[(df['view_label'] == 'PLAX') | (df['view_label'] == 'PSAX')]
        
        # split selection
        if split in ["train", "val", "test"]:
            df = df[df["diagnosis_classifier_split"]==split]
        elif split != "all":
            assert ValueError(f"Split must be train/val/test/all, received {split}")
            
        # filter out any labels based on label scheme
        self.scheme = tmed_label_schemes[label_scheme_name]
        df = df[df["diagnosis_label"].isin(self.scheme.keys())]
            
        # populate some fields useful for input/label reading
        df['path'] = df.apply(self.get_filename, axis=1)
        
        self.dataset = df
        
        # create one-hot pseudolabels
        self.num_classes = len(np.unique(list(self.scheme.values())))
        self.pseudo = {}
        for i in range(len(self.dataset)):
            data_info = self.dataset.iloc[i]
            label = int(self.scheme[data_info["diagnosis_label"]])
            uid = data_info["path"]
            self.pseudo[uid] = np.zeros(self.num_classes)
            self.pseudo[uid][label] = 1.0
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Accesses an image in the TMED dataset
        Args:
            index (int): 0-to-len(data) index
        Returns:
            'x': image
            'y': original integer label
            'y_u': smoothed non_integer label ndarray of shape C
            'uid': string-based unique identifier for the training sample
        """
        data_info = self.dataset.iloc[index]
        uid = data_info['path']
        img = load_image(uid) # 112x112
        y = int(self.scheme[data_info["diagnosis_label"]]) # human-assigned GT
        y_u = self.pseudo[uid] # uncertainty-augmented target
        view = data_info['view_label']
        y_view = int(tmed_view_schemes['all'][view])
        n_view_classes = len(np.unique(list(tmed_view_schemes['all'].values())))
        y_view_u = np.zeros(n_view_classes)
        y_view_u[y_view] = 1.0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'x':img, 'y':y, 'y_u':y_u, 'uid':uid, 'view':view, 'query_key':data_info['query_key']}

    def __len__(self):
        return len(self.dataset)
        
    def get_filename(self, row):
        return os.path.join(self.dataset_root, row['SourceFolder'], row['query_key'])
    
    def get_pseudo(self):
        return self.pseudo
        
    def set_pseudo(self, new_pseudo):
        keys_not_found = []
        for k in new_pseudo.keys():
            if self.pseudo.get(k) is not None:
                self.pseudo[k] = new_pseudo[k]
            else:
                keys_not_found.append(k)
        if len(keys_not_found) > 0:
            print(f"Warning: new keys {k} do not exist in existing uid set")
            
    def class_sampler(self):
        """
        returns sampler (WeightedRandomSamplers) based on frequency of the AS classes occurring
        """
        numerical_labels = self.dataset["diagnosis_label"].apply(lambda x: self.scheme[x])
        class_sample_count_as = numerical_labels.value_counts()[range(self.num_classes)].to_numpy()
        print(class_sample_count_as)
        weight_as = 1.0 / class_sample_count_as
        print(weight_as)
        samples_weight_as = weight_as[numerical_labels.to_numpy()]
        sampler_as = WeightedRandomSampler(samples_weight_as, len(samples_weight_as))
        return sampler_as
        
if __name__ == "__main__":
    data_config = {"batch_size":1, "data_root":"/data/TMED/approved_users_only", "img_resolution":224, "label_scheme_name":"tufts"}
    dm = TMED2_DataModule(**data_config)
    stage = "fit"
    dm.setup(stage)
    dm.ds_train.class_sampler()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    means = []
    gammas = []
    for i, data in enumerate(train_loader):
        if i == 0:
            print(data['x'].shape)
            print(data['x'])
            print(data['y'])
            print(data['y_u'])
            print(data['uid'])
            print(data['view'])
        # # compute the gamma for the image
        # mid = 0.5
        # mean = np.mean(data['x'].numpy() * 255)
        # gamma = np.log(mid*255)/np.log(mean)
        # gammas.append(gamma)
        means.append(np.mean(data['x'].numpy()))
    std = np.std(means)
    print(np.mean(means))
    print(std)
    # print(np.mean(gammas))
    # print(np.std(gammas))
    # print(gammas)
    