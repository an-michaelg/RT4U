# PTL datamodule of AS dataset

import os
from os.path import join
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from random import randint, uniform
from scipy.io import loadmat
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import RandomResizedCropVideo
import lightning.pytorch as pl

from as_tom_data_utils import label_schemes, compute_intervals
from data_transforms import RandomRotateVideo

DATA_ROOT = "D:/Datasets/aorticstenosis/round2"
CSV_NAME = "D:/Datasets/aorticstenosis/round2/annotations-all.csv"


class ASDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = DATA_ROOT,
        csv_name: str = CSV_NAME,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: str = "random",  # one of 'AS', 'random'
        label_scheme_name: str = "all",  # see as_tom_data_utils.label_schemes
        view: str = "plax",  # one of  psax, plax, all
        iterate_intervals: bool = True,  # true to get multiple images/videos from the same dicom in sequence during inference
        interval_unit: str = "cycle",  # get X number of image/second/cycles
        interval_quant: float = 1.1,
        augmentation: bool = True,
        transform_rotate_degrees: int = 15,
        transform_min_crop_ratio: float = 0.7,
        transform_time_dilation: float = 0.2,  # relative temporal offset to the interval taken, only applies if interval_unit != image and iterate_intervals=False
        normalize: bool = True,
        img_resolution: int = 224,
        frames: int = 16,  # 1 for image-based, 2 or more for video-based
        **kwargs,
    ):
        super().__init__()

        # navigation
        self.data_dir = data_dir
        self.csv_name = csv_name
        # dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler = sampler
        # subset of dataset
        self.label_scheme_name = label_scheme_name
        self.view = view
        # video quantity per sample
        self.iterate_intervals = iterate_intervals
        self.interval_unit = interval_unit
        self.interval_quant = interval_quant
        # transformation
        self.augmentation = augmentation
        self.transform_rotate_degrees = transform_rotate_degrees
        self.transform_min_crop_ratio = transform_min_crop_ratio
        self.transform_time_dilation = transform_time_dilation
        self.normalize = normalize
        # output size
        self.img_resolution = img_resolution
        self.frames = frames

    def setup(self, stage: str):
        self.dset_train = self.get_AS_dataset(split="train", mode="train")
        self.dset_val = self.get_AS_dataset(split="val", mode="val")
        self.dset_test = self.get_AS_dataset(split="val", mode="test")
        self.dset_predict = self.get_AS_dataset(split="val", mode="test")

    def get_AS_dataset(self, split="train", mode="train"):
        dset = AorticStenosisDataset(
            data_info_file=self.csv_name,
            dataset_root=self.data_dir,
            view=self.view,
            split=split,
            label_scheme_name=self.label_scheme_name,
            interval_iteration=(mode != "train"),
            interval_add_offset=False,
            interval_unit=self.interval_unit,
            interval_quant=self.interval_quant,
            transform=(mode == "train"),
            transform_rotate_degrees=self.transform_rotate_degrees,
            transform_min_crop_ratio=self.transform_min_crop_ratio,
            transform_time_dilation=self.transform_time_dilation,
            normalize=self.normalize,
            frames=self.frames,
            img_size=self.img_resolution,
            return_info=False,  # (mode == "test"),
        )
        return dset

    def train_dataloader(self):
        if self.sampler == "AS":
            sampler_AS = self.dset_train.class_sampler_AS()
            return DataLoader(
                self.dset_train,
                batch_size=self.batch_size,
                sampler=sampler_AS,
                num_workers=self.num_workers,
            )
        else:
            return DataLoader(
                self.dset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(
            self.dset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dset_test, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dset_predict, batch_size=1, shuffle=False, num_workers=self.num_workers
        )
        
    def get_pseudo(self):
        return self.ds_train.get_pseudo()
    
    def set_pseudo(self, pseudo):
        # modify the pseudo property of ds_train
        self.ds_train.set_pseudo(pseudo)


class AorticStenosisDataset(Dataset):
    def __init__(
        self,
        data_info_file=CSV_NAME,
        dataset_root: str = DATA_ROOT,
        view: str = "plax",
        split: str = "train",
        sample_size=None,  # to load only the first "sample_size" cases of the dataframe (for quick epoch runs)
        label_scheme_name: str = "all",
        transform: bool = False,
        transform_rotate_degrees: float = 10.0,
        transform_min_crop_ratio: float = 0.7,
        transform_time_dilation: float = 0.2,
        normalize: bool = False,
        frames: int = 16,
        img_size: int = 224,
        return_info: bool = False,  # Whether to return all metadata for clips, should be used sparingly
        interval_iteration: bool = False,  # False = one 'unit' per video, True = multiple 'units' in sequence
        interval_add_offset: bool = False,  # introduce random offsets to iterative mode video loading
        interval_unit: str = "cycle",  # image/second/cycle = get X images/seconds/cycles
        interval_quant: float = 1.0,  # X in images/seconds/cycles
        **kwargs,
    ):
        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(data_info_file)
        dataset = dataset.rename(columns={"Unnamed: 0": "db_indx"})
        # append dataset root to each path in the dataframe
        # tip: map(lambda x: x+1) means add 1 to each element in the column
        dataset["path"] = dataset["path"].map(lambda x: join(dataset_root, Path(x)))

        ##### VIEW, LABEL AND SPLIT SUB-SET SELECTION #####
        if view in ("plax", "psax"):
            dataset = dataset[dataset["view"] == view]
        elif view != "all":
            raise ValueError(f"View should be plax/psax/all, got {view}")

        # remove unnecessary columns in 'as_label' based on label scheme
        self.scheme = label_schemes[label_scheme_name]
        dataset = dataset[dataset["as_label"].isin(self.scheme.keys())]

        self.return_info = return_info
        # self.hr_mean, self.hr_std = 4.237, 0.1885

        # Take train/test/val
        if split in ("train", "val", "test"):
            dataset = dataset[dataset["split"] == split]
        elif split != "all":
            raise ValueError(f"View should be train/val/test/all, got {split}")

        if sample_size is not None:
            dataset = dataset.sample(sample_size)

        ##### CALCULATE NUMBER OF CLIPS TO LOAD BASED ON VIDEO LENGTHS #####
        # check the number of image/sub-video intervals we can get per video
        # and create a mapping between dataset entries and intervals
        self.interval_iteration = interval_iteration
        self.interval_unit = interval_unit
        self.interval_quant = interval_quant
        if frames == 1:
            assert (
                interval_unit == "image"
            ), "For drawing 1 frame from dataloader, interval_unit must be image"
            assert (
                frames == interval_quant
            ), "For drawing 1 frame from dataloader, interval_quant must also be 1"
        if self.interval_iteration:
            dataset, dataset_intervals = compute_intervals(
                dataset, interval_unit, interval_quant, interval_add_offset
            )
            self.dataset_intervals = dataset_intervals
        else:
            dataset, _ = compute_intervals(
                dataset, interval_unit, interval_quant, interval_add_offset
            )

        self.dataset = dataset

        ##### CONFIGURE INPUT DIMENSIONALITY #####
        self.frames = frames
        self.resolution = (img_size, img_size)

        ##### CONFIGURE TRANSFORMS #####
        self.transform = None
        self.transform_time_dilation = 0.0
        if transform:
            self.transform = Compose(
                [
                    RandomResizedCropVideo(
                        size=self.resolution, scale=(transform_min_crop_ratio, 1)
                    ),
                    RandomRotateVideo(degrees=transform_rotate_degrees),
                ]
            )
            self.transform_time_dilation = transform_time_dilation
        self.normalize = normalize
        
        ##### CONFIGURE PSEUDOLABELS #####
        self.num_classes = len(np.unique(list(self.scheme.values())))
        self.pseudo = {}
        for i in range(len(self.dataset)):
            data_info = self.dataset.iloc[i]
            label = int(self.scheme[data_info["as_label"]])
            uid = data_info["path"]
            self.pseudo[uid] = np.zeros(self.num_classes)
            self.pseudo[uid][label] = 1.0
            
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

    def class_sampler_AS(self):
        """
        returns samplers (WeightedRandomSamplers) based on frequency of the AS class occurring
        """
        labels_as = self.dataset.apply(lambda x: self.scheme[x.as_label], axis=1).values
        class_sample_count_as = self.dataset.as_label.value_counts()[
            self.scheme.keys()
        ].to_numpy()
        weight_as = 1.0 / class_sample_count_as
        samples_weight_as = weight_as[labels_as]
        sampler_as = WeightedRandomSampler(samples_weight_as, len(samples_weight_as))
        return sampler_as

    def __len__(self) -> int:
        """
        iterative interval mode uses an "expanded" version of the dataset
        where each interval of a video can be considered as a separate data
        instance, thus the length of the dataset depends on the interval mode
        """
        if self.interval_iteration:
            return len(self.dataset_intervals)
        else:
            return len(self.dataset)

    @staticmethod
    def get_random_interval(vid_length, length):
        if length > vid_length:
            return 0, vid_length
        else:
            start = randint(0, vid_length - length)
            return start, start + length

    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1, -1)

    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        """
        normalizes the input tensor
        :param in_tensor: needs to be already in range of [0,1]
        """
        # in_tensor is 1xTxHxW
        m = 0.099
        std = 0.171
        return (in_tensor - m) / std

    def _get_item_from_info(self, data_info, window_start, window_end, interval_idx=0):
        """
        General method to get an image and apply tensor transformation to it

        Parameters
        ----------
        data_info : pd.DataFrame
            pdSeries (can be thought of as 1-row dataframe) of item to retrieve
        window_start : int
            frame of the start of the interval to retrieve
        window_end : int
            frame of the end of the interval to retrieve (non-inclusive)
        interval_idx : int
            index of the interval within the cine (eg. first interval is 0)

        Returns
        -------
        ret : 3xTxHxW tensor (if video) 3xHxW tensor (if image)
            representing image/video w standardized dimensionality
            if return_info is true, also returns additional information

        """
        cine_original = loadmat(data_info["path"])["cine"]  # T_original xHxW
        cine = cine_original[window_start:window_end]  # T_window xHxW

        cine = resize(
            cine, (self.frames, *self.resolution)
        )  # TxHxW, where T=1 if image, Note range becomes [0,1] here
        cine = torch.tensor(cine).unsqueeze(0)  # 1xTxHxW, where T=1 if image

        label_as = torch.tensor(self.scheme[data_info["as_label"]])
        label_as_soft = self.pseudo[data_info["path"]]
        view = torch.tensor((data_info["view"] == "psax") * 1)

        if self.transform:
            cine = self.transform(cine)
        if self.normalize:
            cine = self.bin_to_norm(cine)
        cine = self.gray_to_gray3(cine)  # shape = (3,T,H,W), where T=1 if image
        cine = cine.float()  # 3xTxHxW, where T=1 if image

        if self.frames == 1:
            cine = cine[:, 0]  # 3xTxHxW (for video), 3xHxW (for image)

        ret = {
            "uid": data_info.path,
            "x": cine,
            "y": label_as,
            "y_u": label_as_soft,
            "view": view,
            "interval_idx": interval_idx,
            "window_start": window_start,
            "window_end": window_end,
            "original_length": cine_original.shape[0],
        }
        if self.return_info:
            di = data_info.to_dict()
            ret.update({"data_info": di, "cine_original": cine_original})
        return ret

    def __getitem__(self, item):
        """
        iterative interval mode uses an "expanded" version of the dataset
        where each interval of a video can be considered as a separate data
        instance, thus the length of the dataset depends on the interval mode
        """
        if self.interval_iteration:
            data_interval = self.dataset_intervals.iloc[item]
            video_id = data_interval["video_idx"]
            data_info = self.dataset.iloc[video_id]
            start_frame = data_interval["start_frame"]
            end_frame = data_interval["end_frame"]
            interval_idx = data_interval["interval_idx"]
        else:
            data_info = self.dataset.iloc[item]
            # determine a random window
            ttd = self.transform_time_dilation
            if self.interval_unit == "image":
                wsize = int(self.interval_quant)
            else:  # can slightly vary the window size
                wsize = max(
                    int(data_info["window_size"] * uniform(1 - ttd, 1 + ttd)), 1
                )
            start_frame, end_frame = self.get_random_interval(
                data_info["frames"], wsize
            )
            interval_idx = 0

        return self._get_item_from_info(data_info, start_frame, end_frame, interval_idx)


if __name__ == "__main__":
    data_config = {
        "data_dir": "/data/datasets/Aortic_Stenosis/as_tom",
        "csv_name": "/data/datasets/Aortic_Stenosis/as_tom/annotations-all.csv",
        "batch_size": 2,
        "num_workers": 0,
        "sampler": "random",  # one of 'AS', 'random'
        "label_scheme_name": "all",  # one of 'binary', 'all', 'not_severe', 'as_only', 'mild_moderate', 'moderate_severe'
        "view": "psax",  # one of  psax, plax, all
        "iterate_intervals": True,
        "interval_unit": "cycle",
        "interval_quant": 1.0,
        "augmentation": True,
        "transform_min_crop_ratio": 0.7,
        "transform_rotate_degrees": 15,
        "normalize": True,
        "img_resolution": 112,
        "frames": 32,
    }
    dm = ASDataModule(**data_config)
    stage = "test"
    dm.setup(stage)
    if stage == "fit":
        dataloader = dm.train_dataloader()
    elif stage == "test":
        dataloader = dm.test_dataloader()

    # dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    # print("len of dataset = {}".format(len(dataset)))
    print("len of dataloader iteration = {}".format(len(dataloader)))
    # ################### Iterate over samples #########################

    data_iter = iter(dataloader)
    sample_dict = next(
        data_iter
    )  # sample shape = (N,3,T,H,W) for video, (N,3,H,W) for image

    print(
        f"target_AS: {sample_dict['target_AS']}\n"
        f"view: {sample_dict['view']}\n"
        f"Label shape : {sample_dict['target_AS'].shape} \n"
        f"filenames : {sample_dict['filename']} \n"
        f"cine shape: {sample_dict['cine'].shape}"
    )
