import argparse
import os
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm, trange

from models.dataset import get_data_loader
from models.fusion_restoration_net import DecoupledDecoder,FusionEncoder
from models.features import MultimodalFeatures
from models.loss import image_similarity_loss, census_loss,smoothness_loss

def set_seeds(sid=115):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

    
def train_CFM(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    # Dataloader
    train_loader = get_data_loader("train", class_name=args.class_name, img_size=224,
                                   dataset_path=args.dataset_path, batch_size=args.batch_size, shuffle=True)

    # Feature extractors
    feature_extractor = MultimodalFeatures()

    # Model instantiation
    fusion_encoder = FusionEncoder(in_features_2D=768, in_features_3D=1152, out_features=960)
    decoder_2D = DecoupledDecoder(in_features=960, out_features=768) 
    decoder_3D = DecoupledDecoder(in_features=960, out_features=1152) 

    optimizer = torch.optim.Adam(params=chain(fusion_encoder.parameters(), decoder_2D.parameters(), decoder_3D.parameters()))

    fusion_encoder.to(device), decoder_2D.to(device), decoder_3D.to(device)

    for epoch in trange(args.epochs_no, desc=f'Training Feature Transfer Net'):
        epoch_total_loss = []

        # ------------ [Training Loop] ------------ #
        for (rgb, pc, _), _ in tqdm(train_loader, desc=f'Extracting feature from class: {args.class_name}', unit='batch', leave=True):
            rgb, pc = rgb.to(device), pc.to(device)

            # Make models trainable
            fusion_encoder.train(), decoder_2D.train(), decoder_3D.train()

            if args.batch_size == 1:
                rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
            else:
                rgb_patches = []
                xyz_patches = []

                for i in range(rgb.shape[0]):
                    rgb_patch, xyz_patch = feature_extractor.get_features_map(rgb[i].unsqueeze(dim=0), pc[i].unsqueeze(dim=0))
                    rgb_patches.append(rgb_patch)
                    xyz_patches.append(xyz_patch)

                rgb_patch = torch.stack(rgb_patches, dim=0)
                xyz_patch = torch.stack(xyz_patches, dim=0)

            # Fusion and Restoration
            fusion_embedding = fusion_encoder(rgb_patch, xyz_patch)
            restored_2D = decoder_2D(fusion_embedding)
            restored_3D = decoder_3D(fusion_embedding)

            # Compute the losses
            loss_sim_2D = image_similarity_loss(rgb_patch, restored_2D)
            loss_sim_3D = image_similarity_loss(xyz_patch, restored_3D)

            # Smoothness loss
            deformation_field_2D = restored_2D - rgb_patch
            deformation_field_3D = restored_3D - xyz_patch
            loss_smooth_2D = smoothness_loss(deformation_field_2D, rgb_patch)
            loss_smooth_3D = smoothness_loss(deformation_field_3D, xyz_patch)

            # Census loss
            loss_census_2D = census_loss(rgb_patch, restored_2D)
            loss_census_3D = census_loss(xyz_patch, restored_3D)

            # Total loss
            total_loss = loss_sim_2D + loss_sim_3D + loss_smooth_2D + loss_smooth_3D + loss_census_2D + loss_census_3D

            epoch_total_loss.append(total_loss.cpu())

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                exit()

            # Optimization
            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        # Print epoch loss
        avg_epoch_loss = torch.Tensor(epoch_total_loss).mean().item()
        tqdm.write(f'Epoch {epoch + 1}/{args.epochs_no}, Loss: {avg_epoch_loss:.4f}')

    # Final model saving
    directory = f'{args.checkpoint_savepath}/{args.class_name}'
    os.makedirs(directory, exist_ok=True)

    torch.save(fusion_encoder.state_dict(), os.path.join(directory, 'fusion_encoder_' + model_name + '.pth'))
    torch.save(decoder_2D.state_dict(), os.path.join(directory, 'decoder_2D_' + model_name + '.pth'))
    torch.save(decoder_3D.state_dict(), os.path.join(directory, 'decoder_3D_' + model_name + '.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Attention Fusion Restoration (AFR) on a dataset.')
    parser.add_argument('--dataset_path', default='./datasets/mvtec3d', type=str, help='Dataset path.')
    parser.add_argument('--checkpoint_savepath', default='./checkpoints/mvtec', type=str, help='Where to save the model checkpoints.')
    parser.add_argument('--class_name', default=None, type=str, choices=["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
                                                                        'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help='Category name.')
    parser.add_argument('--epochs_no', default=100, type=int, help='Number of epochs to train the CFMs.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch dimension. Usually 16 is around the max.')
    args = parser.parse_args()
    train_CFM(args)