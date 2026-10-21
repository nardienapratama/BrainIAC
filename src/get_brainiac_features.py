import os
# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
from dataset import QuadImageMGMTDataset
from load_brainiac import load_brainiac

# fix random seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda_avail = torch.cuda.is_available()

print(f"CUDA availability: {cuda_avail}")
print(f"Selected device: {device}")

if cuda_avail:
    torch.cuda.manual_seed_all(seed)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

#=========================
# Inference function
#=========================

def infer(model, test_loader):
    features_df = []  # Placeholder for feature DataFrame
    model.eval()
    
    # with torch.no_grad():
    with torch.inference_mode():
        mri_types = ['t1', 't1ce', 't2', 'flair']
        print("Waiting for first batch...", flush=True)

        for sample in tqdm(test_loader, desc="Extracting ViT features", unit="batch"):
            if sample is None:
                continue
            else:
                print("Going through samples...")
                # process for all modalities

                # NEW VERSION - ALL PATIENTS DONE AT THE SAME TIME

                # if mri types are in the feature columns
                if not all(mri_type in sample for mri_type in mri_types):
                    continue
                
                batch_inputs = torch.stack(
                    [
                        # [[1,96,96,96], [1,96,96,96], ...] for n batch size 
                        # stack -> [1,96,96,96] from patient A x 4 modalities
                        # -> [4, 1, 96, 96, 96] which is:
                        # [ 
                        #   [ pat 1 - T1],
                        #   [ pat 1 - T1CE],
                        #   ...
                        # ] for other modalities...
                        torch.stack(
                            [sample[mri_type][i] for mri_type in mri_types],
                            dim=0,
                        )
                        for i in range(len(sample["t1"]))
                    ],
                    # stack [4, 1, 96, 96, 96] for n patients (batch size)
                    # [n, 4, 1, 96, 96, 96]
                    dim=0
                ).to(device, non_blocking=True) # stack -> [n, 1, 96, 96, 96] -> [n, 4, 1, 96, 96, 96]


                # batch inputs shape: batch size, 4, channels, dept, height, width
                batch_size, num_mri_mods, channels, depth, height, width = batch_inputs.shape

                inputs = torch.reshape(batch_inputs, (batch_size * num_mri_mods,channels,depth,height,width))
                

                # Get features from the ViT backbone model
                features = model(inputs)


                ##### SHAPE CHECK
                if features.ndim != 2:
                    raise RuntimeError(
                        f"Expected model output with 2 dimensions, got {features.shape}"
                    )

                expected_rows = batch_size * num_mri_mods

                if features.shape[0] != expected_rows:
                    raise RuntimeError(
                        f"Expected {expected_rows} rows, but got {features.shape}!"
                    )

                features = features.reshape(batch_size, num_mri_mods, -1) # [batch size, 4, 768]

                features_numpy = features.cpu().numpy()

                labels_numpy = (
                                    torch.stack(sample["label"])
                                    .cpu()
                                    .numpy()
                                    .flatten()
                                ) # shape: (batch size,)

                batch_size, num_mri_mods, feature_size = features_numpy.shape


                # Expand features into separate columns
                feature_columns = [f'Feature_{i}' for i in range(features_numpy.shape[2])]

                batch_features = pd.DataFrame(features_numpy.reshape(batch_size*num_mri_mods, -1), columns=feature_columns)


                # Add label column (flatten to 1D)
                batch_features['GroundTruthClassLabel'] = np.repeat(labels_numpy, num_mri_mods) # (batch size,) -> (batch size*4,)
                batch_features['modality'] = mri_types * batch_size # [t1, t1ce, t2, flair] -> [t1, t1ce, t2, flair,t1, t1ce, t2, flair,t1, t1ce, t2, flair,t1, t1ce, t2, flair]
                batch_features['pat_id'] = np.repeat(sample['pat_id'], num_mri_mods) # (batch size,) -> (batch size*4,)

                
                features_df.append(batch_features)

    # After processing all modalities for the current sample, append temp_df to features_df
    if not features_df:
        return None
    features_df = pd.concat(features_df, ignore_index=True)
    
    return features_df

#=========================
# Main inference pipeline
#=========================

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Extract ViT BrainIAC features from images')
    parser.add_argument('--checkpoint', type=str, default=None,required=True,
                      help='Path to the ViT BrainIAC model checkpoint (default: checkpoints/BrainIAC.ckpt)')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='Path to the input CSV file containing image paths')
    parser.add_argument('--output_csv', type=str, required=True,
                      help='Path to save the output features CSV')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing the image data')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers for data loading (default: 1)')
    args = parser.parse_args()

    # Setup dataset with validation transforms (no augmentation for feature extraction)
    # test_dataset = BrainAgeDataset(
    #     csv_path=args.input_csv,
    #     root_dir=args.root_dir,
    #     transform=get_validation_transform()
    # )
    
    # test_dataset = BrainImageDataset(
    #     csv_path=args.input_csv,
    #     root_dir=args.root_dir,
    #     transform=get_validation_transform()
    # )
    print("Creating test dataset...")
    test_dataset = QuadImageMGMTDataset(
        csv_path=args.input_csv,
        root_dir=args.root_dir
    )

    if len(test_dataset) == 0:
        print(f"Skipping {args.root_dir} (patient not in CSV)")
        return

    def collate_remove_none(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0 or not batch:
            return None
        return {k:  [d[k] for d in batch] for k in batch[0].keys()}

    print("Creating test dataloader...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
        collate_fn=collate_remove_none
    )

    # Load ViT BrainIAC - device placement is handled in load_brainiac
    model = load_brainiac(args.checkpoint, device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Extract features
    print("Performing inference...")
    features_df = infer(model, test_loader)
    if features_df is None:
        print("No features were extracted.")
        return

    # Save features
    features_df.to_csv(args.output_csv, index=False)
    print(f"ViT BrainIAC features saved to {args.output_csv}")
    print(f"Feature shape: {features_df.shape}")
    print(f"Number of feature dimensions: {features_df.shape[1] - 3}")  # -1 for label column, modality column, patient ID

if __name__ == "__main__":
    main()