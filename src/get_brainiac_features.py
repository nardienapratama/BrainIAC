import torch
import numpy as np
import pandas as pd
import random
import yaml
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import BrainImageDataset, get_validation_transform
from load_brainiac import load_brainiac

# fix random seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#=========================
# Inference function
#=========================

def infer(model, test_loader):
    features_df = None  # Placeholder for feature DataFrame
    model.eval()
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Extracting ViT features", unit="batch"):
            if not sample:  # skip empty batches
                continue
            else:
                # process for all modalities
                temp_df = None

                inputs = torch.stack(sample['image']).to(device)
                class_labels = torch.stack(sample['label']).float().to(device)

                # Get features from the ViT backbone model
                features = model(inputs)

                features_numpy = features.cpu().numpy()

                # Expand features into separate columns
                feature_columns = [f'Feature_{i}' for i in range(features_numpy.shape[1])]
                batch_features = pd.DataFrame(
                    features_numpy,
                    columns=feature_columns
                )

                # Add metadata columns (ensure correct length)
                # If batch size is 1, these will be lists of length 1
                for meta_key in ['pat_id', 'modality', 'collection']:
                    batch_features[meta_key] = sample[meta_key]

                # Add label column (flatten to 1D)
                batch_features['GroundTruthClassLabel'] = class_labels.cpu().numpy().flatten()

                # Append batch features to temp_df
                if temp_df is None:
                    temp_df = batch_features
                else:
                    temp_df = pd.concat([temp_df, batch_features], ignore_index=True)
                # After processing all modalities for the current sample, append temp_df to features_df
                if features_df is None:
                    features_df = temp_df
                else:
                    features_df = pd.concat([features_df, temp_df], ignore_index=True)
    
    return features_df

#=========================
# Main inference pipeline
#=========================

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Extract ViT BrainIAC features from images')
    parser.add_argument('--checkpoint', type=str, default=None,
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
    
    test_dataset = BrainImageDataset(
        csv_path=args.input_csv,
        root_dir=args.root_dir,
        transform=get_validation_transform()
    )

    if len(test_dataset) == 0:
        print(f"Skipping {args.root_dir} (patient not in CSV)")
        return

    def collate_remove_none(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return {k:  [d[k] for d in batch] for k in batch[0].keys()}

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_remove_none
    )

    # Load ViT BrainIAC - device placement is handled in load_brainiac
    model = load_brainiac(args.checkpoint, device)
    
    # Extract features
    features_df = infer(model, test_loader)
    
    # Save features
    features_df.to_csv(args.output_csv, index=False)
    print(f"ViT BrainIAC features saved to {args.output_csv}")
    print(f"Feature shape: {features_df.shape}")
    print(f"Number of feature dimensions: {features_df.shape[1] - 1}")  # -1 for label column

if __name__ == "__main__":
    main()