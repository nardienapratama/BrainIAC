import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd,
    NormalizeIntensityd,
    RandAffined, RandFlipd, RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd, ToTensord
)

def get_default_transform(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandAffined(
            keys=["image"],
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.5,
            padding_mode="border"
        ),
        # Only left-right flipping (typically axis 2 for brain MRI in RAS orientation)
        # This preserves anatomical correctness while providing useful augmentation
        RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),  # Left-right flip only
        #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3)),
        
        ToTensord(keys=["image"])
    ])

def get_validation_transform(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ToTensord(keys=["image"])
    ])

class BrainAgeDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transform()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']  # Regression value for stroke/MCI
        #dataset = str(self.dataframe.loc[idx, 'dataset'])
        
        # Construct image path for MCI/Stroke format
        img_path = os.path.join(self.root_dir,  pat_id  + ".nii.gz")
        sample = {"image": img_path}
        sample = self.transform(sample)
        return {"image": sample["image"], "label": torch.tensor(label, dtype=torch.float32)}


class BrainImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.root_dir = root_dir
        # check if root dir contains folder for each image id, if so, we will extract the image id
        self.dataframe = pd.read_csv(csv_path, dtype={"collection": str, "pat_id": str, "modality":str, "label": float})
        print(self.root_dir.split("/")[-1])
        print(self.dataframe['pat_id'].values)
        print("is value there: ", self.root_dir.split("/")[-1] in self.dataframe['pat_id'].values)
        current_patient_id = self.root_dir.split("/")[-1]
        if current_patient_id in self.dataframe['pat_id'].values:
            self.dataframe = self.dataframe[self.dataframe["pat_id"] == current_patient_id].reset_index(drop=True)
        else:
            self.dataframe = self.dataframe.iloc[0:0]
        self.transform = transform if transform is not None else get_default_transform()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # retrieve all modality file paths for current image id if self.current_patient_id is set, otherwise retrieve single modality file path as before
        # if hasattr(self, 'current_patient_id'):
        #     pat_id = self.current_patient_id
        #     #retrieve collection and modality for current patient id from dataframe
        #     collection = str(self.dataframe[self.dataframe['pat_id'] == pat_id]['collection'].values[0])
        #     modality = str(self.dataframe[self.dataframe['pat_id'] == pat_id]['modality'].values[0])
        #     label = float(self.dataframe[self.dataframe['pat_id'] == pat_id]['label'].values[0])
        #     results = []
        #     for mod in modality:
        #         # Construct image path for current patient id, collection, and modality
        #         img_path = os.path.join(self.root_dir, pat_id + "_" + mod + "_0000" + ".nii.gz")
        #         sample = {"image": img_path}
        #         sample = self.transform(sample)
        #         result = {"image": sample["image"],
        #         "collection": collection,
        #         "pat_id": pat_id,
        #         "modality": modality,
        #         "label": torch.tensor(label, dtype=torch.float32)}
        #         results.append(result)

        # else:
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])

        collection = str(self.dataframe.loc[idx, 'collection'])
        modality = str(self.dataframe.loc[idx, 'modality'])  # modality label (e.g. t1ce, t2, etc.)
        label = float(self.dataframe.loc[idx, 'label'])
    
        img_path = os.path.join(self.root_dir,  pat_id  + "_" + modality + ".nii.gz")
        print("Getting image from:", img_path, ".......................")
        if not os.path.exists(img_path):
            return None  # Return empty list if image file does not exist
        sample = {"image": img_path}
        sample = self.transform(sample)
        results = {"image": torch.tensor(sample["image"], dtype=torch.float32),
                "collection": collection,
                "pat_id": pat_id,
                "modality": modality,
                "label": torch.tensor(label, dtype=torch.float32)}
        return results


class MCIStrokeDataset(Dataset):
    """Dataset class for MCI and Stroke tasks"""
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transform()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']  # Regression value for stroke/MCI
        #dataset = str(self.dataframe.loc[idx, 'dataset'])
        
        # Construct image path for MCI/Stroke format
        img_path = os.path.join(self.root_dir, pat_id + ".nii.gz")
        sample = {"image": img_path}
        sample = self.transform(sample)
        return {"image": sample["image"], "label": torch.tensor(label, dtype=torch.float32)}


class SequenceDataset(Dataset):
    """Dataset class for Sequence multiclass classification"""
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"PatientID": str, "SequenceLabel": str, "ScanID": str, "Sequence": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transform()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'PatientID'])
        label = self.dataframe.loc[idx, 'SequenceLabel']
        label  = int(label) - 1
        scan_id = self.dataframe.loc[idx, 'ScanID']
        sequence = self.dataframe.loc[idx, 'Sequence']
        dataset = str(self.dataframe.loc[idx, 'Dataset'])
        #img_path = os.path.join(self.root_dir + "/" + dataset + "/data", scan_id + ".nii.gz" )
        img_path = os.path.join(self.root_dir + "/" + dataset + "/data", pat_id + "-" + scan_id + "-" + sequence + ".nii.gz" )
        sample = {"image": img_path}
        sample = self.transform(sample)
        return {"image": sample["image"], "label": torch.tensor(label, dtype=torch.float32)} 


#==============================================
## DATASET CLASS AND TRANSFORMS FOR MULTI SEQUENCE INPUTS
#==============================================







def get_default_transform_dual(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image1", "image2"]),
        EnsureChannelFirstd(keys=["image1", "image2"]),
        Resized(keys=["image1", "image2"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image1", "image2"], nonzero=True, channel_wise=True),
        RandAffined(
            keys=["image1", "image2"],
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.5,
            padding_mode="border"
        ),
        # Only left-right flipping (typically axis 2 for brain MRI in RAS orientation)
        # This preserves anatomical correctness while providing useful augmentation
        RandFlipd(keys=["image1", "image2"], spatial_axis=[2], prob=0.5),  # Left-right flip only
        #ScaleIntensityd(keys=["image1", "image2"], minv=0.0, maxv=1.0),
        RandGaussianSmoothd(keys=["image1", "image2"], prob=0.2),
        RandGaussianNoised(keys=["image1", "image2"], prob=0.2, std=0.05),
        RandAdjustContrastd(keys=["image1", "image2"], prob=0.2, gamma=(0.7, 1.3)),
        ToTensord(keys=["image1", "image2"])
    ])

def get_validation_transform_dual(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image1", "image2"]),
        EnsureChannelFirstd(keys=["image1", "image2"]),
        Resized(keys=["image1", "image2"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image1", "image2"], nonzero=True, channel_wise=True),
        #ScaleIntensityd(keys=["image1", "image2"], minv=0.0, maxv=1.0),
        ToTensord(keys=["image1", "image2"])
    ])

def dual_image_collate_fn(batch):
    """
    Collate function for DualImageDataset to stack image1 and image2 into a single tensor.
    Args:
        batch: List of dicts with keys 'image1', 'image2', 'label'.
    Returns:
        images: Tensor of shape (batch_size, 2, C, D, H, W)
        labels: Tensor of shape (batch_size,)
    """
    images1 = [item['image1'] for item in batch]
    images2 = [item['image2'] for item in batch]
    # Each image1/image2 is (C, D, H, W), stack to (batch, C, D, H, W)
    images1 = torch.stack(images1, dim=0)
    images2 = torch.stack(images2, dim=0)
    # Stack along new dim=1 to get (batch, 2, C, D, H, W)
    images = torch.stack([images1, images2], dim=1)
    labels = torch.stack([item['label'] for item in batch], dim=0)
    return images, labels

class DualImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transform_dual()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']
        
        # Construct paths for both image modalities
        img_path1 = os.path.join(self.root_dir, pat_id + "_t2f.nii.gz")
        img_path2 = os.path.join(self.root_dir, pat_id + "_t1ce.nii.gz")
    
        sample = {"image1": img_path1, "image2": img_path2}
        
        if self.transform:
            sample = self.transform(sample)

        # Return as before, but note that collate_fn will stack these for model input
        return {"image1": sample["image1"], "image2": sample["image2"], "label": torch.tensor(label, dtype=torch.float)} 




#==============================================
## DATASET CLASS AND TRANSFORMS FOR QUAD SEQUENCE INPUTS (4 IMAGES)
#==============================================

def get_default_transform_quad(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image1", "image2", "image3", "image4"]),
        EnsureChannelFirstd(keys=["image1", "image2", "image3", "image4"]),
        Resized(keys=["image1", "image2", "image3", "image4"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image1", "image2", "image3", "image4"], nonzero=True, channel_wise=True),
        RandAffined(
            keys=["image1", "image2", "image3", "image4"],
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.5,
            padding_mode="border"
        ),
        # Only left-right flipping (typically axis 2 for brain MRI in RAS orientation)
        # This preserves anatomical correctness while providing useful augmentation
        RandFlipd(keys=["image1", "image2", "image3", "image4"], spatial_axis=[2], prob=0.5),  # Left-right flip only
        #ScaleIntensityd(keys=["image1", "image2", "image3", "image4"], minv=0.0, maxv=1.0),
        RandGaussianSmoothd(keys=["image1", "image2", "image3", "image4"], prob=0.2),
        RandGaussianNoised(keys=["image1", "image2", "image3", "image4"], prob=0.2, std=0.05),
        RandAdjustContrastd(keys=["image1", "image2", "image3", "image4"], prob=0.2, gamma=(0.7, 1.3)),
        ToTensord(keys=["image1", "image2", "image3", "image4"])
    ])

def get_validation_transform_quad(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image1", "image2", "image3", "image4"]),
        EnsureChannelFirstd(keys=["image1", "image2", "image3", "image4"]),
        Resized(keys=["image1", "image2", "image3", "image4"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image1", "image2", "image3", "image4"], nonzero=True, channel_wise=True),
        #ScaleIntensityd(keys=["image1", "image2", "image3", "image4"], minv=0.0, maxv=1.0),
        ToTensord(keys=["image1", "image2", "image3", "image4"])
    ])

def quad_image_collate_fn(batch):
    """
    Collate function for QuadImageDataset to stack image1, image2, image3, image4 into a single tensor.
    Args:
        batch: List of dicts with keys 'image1', 'image2', 'image3', 'image4', 'label'.
    Returns:
        images: Tensor of shape (batch_size, 4, C, D, H, W)
        labels: Tensor of shape (batch_size,)
    """
    images1 = [item['image1'] for item in batch]
    images2 = [item['image2'] for item in batch]
    images3 = [item['image3'] for item in batch]
    images4 = [item['image4'] for item in batch]
    
    # Each image is (C, D, H, W), stack to (batch, C, D, H, W)
    images1 = torch.stack(images1, dim=0)
    images2 = torch.stack(images2, dim=0)
    images3 = torch.stack(images3, dim=0)
    images4 = torch.stack(images4, dim=0)
    
    # Stack along new dim=1 to get (batch, 4, C, D, H, W)
    images = torch.stack([images1, images2, images3, images4], dim=1)
    labels = torch.stack([item['label'] for item in batch], dim=0)
    return images, labels

class QuadImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transform_quad()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'survival']
        #dataset = str(self.dataframe.loc[idx, 'dataset'])
        
        # Construct paths for all four image modalities
        # Common brain tumor sequences: T1, T1c, T2, FLAIR
        img_path1 = os.path.join(self.root_dir,  pat_id + "_t1ce.nii.gz")
        img_path2 = os.path.join(self.root_dir, pat_id + "_t1n.nii.gz")
        img_path3 = os.path.join(self.root_dir, pat_id + "_t2w.nii.gz")
        img_path4 = os.path.join(self.root_dir, pat_id + "_t2f.nii.gz")

    
        sample = {"image1": img_path1, "image2": img_path2, "image3": img_path3, "image4": img_path4}
        
        if self.transform:
            sample = self.transform(sample)

        # Return all four images, collate_fn will stack them for model input
        return {
            "image1": sample["image1"], 
            "image2": sample["image2"], 
            "image3": sample["image3"], 
            "image4": sample["image4"], 
            "label": torch.tensor(label, dtype=torch.float)
        } 

#==============================================
## DATASET CLASS FOR QUAD IMAGE MGMT MODEL 
#==============================================
class QuadImageMGMTDataset(QuadImageDataset):
    def __init__(self, csv_path, root_dir, transform=None):
        super().__init__(csv_path, root_dir, transform)
        # root dir should be brats2021 training set directory
        self.dataframe = pd.read_csv(csv_path, dtype={"collection": str, "pat_id": str, "modality":str, "label": float, "image_path":str})

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']
        modality = str(self.dataframe.loc[idx, 'modality'])
        current_img_path = self.dataframe.loc[idx, 'image_path']
        
        # Construct paths for all four image modalities
        # Common brain tumor sequences: T1, T1c, T2, FLAIR
        img_path1 = os.path.join(current_img_path,  pat_id + "_t1.nii.gz")
        img_path2 = os.path.join(current_img_path, pat_id + "_t1ce.nii.gz")
        img_path3 = os.path.join(current_img_path, pat_id + "_t2.nii.gz")
        img_path4 = os.path.join(current_img_path, pat_id + "_flair.nii.gz")

    
        sample = {"image1": img_path1, "image2": img_path2, "image3": img_path3, "image4": img_path4}
        
        if self.transform:
            sample = self.transform(sample)

        # Return all four images, collate_fn will stack them for model input
        return {
            "t1": sample["image1"], 
            "t1ce": sample["image2"], 
            "t2": sample["image3"], 
            "flair": sample["image4"], 
            "label": torch.tensor(label, dtype=torch.float),
            "pat_id": pat_id,
            "modality": modality
        } 

#==============================================
## DATASET CLASS FOR SEGMENTATION MODEL FINETUNE
#==============================================


def get_default_transform_segmentation(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Resized(keys=["image", "mask"], spatial_size=image_size, mode=["trilinear", "nearest"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandAffined(
            keys=["image", "mask"],
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.5,
            padding_mode="border",
            mode=["bilinear", "nearest"]
        ),
        RandFlipd(keys=["image", "mask"], spatial_axis=[2], prob=0.5),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3)),
        ToTensord(keys=["image", "mask"])
    ])

def get_validation_transform_segmentation(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Resized(keys=["image", "mask"], spatial_size=image_size, mode=["trilinear", "nearest"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "mask"])
    ])

class SegmentationDataset(Dataset):
    """Dataset class for segmentation tasks"""
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transform_segmentation()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        dataset = str(self.dataframe.loc[idx, 'dataset'])
        
        # Construct paths for image and mask
        img_path = os.path.join(self.root_dir, dataset, "FLAIR", pat_id )
        mask_path = os.path.join(self.root_dir, dataset, "singlelabel_seg", pat_id.replace("FLAIR.nii.gz", "tumor_segmentation.nii.gz"))
        
        sample = {"image": img_path, "mask": mask_path}
        
        if self.transform:
            sample = self.transform(sample)
            
        return {
            "image": sample["image"],
            "mask": sample["mask"].long()
        }
