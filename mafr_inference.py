import argparse
import os
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from utils.metrics_utils import calculate_au_pro
from models.fusion_restoration_net import DecoupledDecoder,FusionEncoder
from sklearn.metrics import roc_auc_score
import torch.nn as nn

# Set random seeds for reproducibility
def set_seeds(sid=42):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

# Inference Function
def infer_CFM(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloaders
    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path)

    # Feature extractors (Teacher Model)
    feature_extractor = MultimodalFeatures()

    # Load the trained models
    fusion_encoder = FusionEncoder(in_features_2D=768, in_features_3D=1152, out_features=960).to(device)
    decoder_2D = DecoupledDecoder(in_features=960, out_features=768).to(device)
    decoder_3D = DecoupledDecoder(in_features=960, out_features=1152).to(device)

    # Load the trained models
    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
    checkpoint_path = f'{args.checkpoint_folder}/{args.class_name}'
    fusion_encoder.load_state_dict(torch.load(os.path.join(checkpoint_path, f'fusion_encoder_{model_name}.pth')))
    decoder_2D.load_state_dict(torch.load(os.path.join(checkpoint_path, f'decoder_2D_{model_name}.pth')))
    decoder_3D.load_state_dict(torch.load(os.path.join(checkpoint_path, f'decoder_3D_{model_name}.pth')))

    fusion_encoder.eval(), decoder_2D.eval(), decoder_3D.eval()

    # Use box filters to approximate Gaussian blur
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u**2)

    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []

    # ------------ [Testing Loop] ------------ #
    for (rgb, pc, depth), gt, label, rgb_path in tqdm(test_loader, desc=f'Extracting feature from class: {args.class_name}.'):
        rgb, pc, depth = rgb.to(device), pc.to(device), depth.to(device)

        with torch.no_grad():
            # Extract features
            rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)

            # Fusion and Restoration
            fusion_embedding = fusion_encoder(rgb_patch, xyz_patch)
            restored_2D = decoder_2D(fusion_embedding)
            restored_3D = decoder_3D(fusion_embedding)

            # Mask for valid 3D points
            xyz_mask = (xyz_patch.sum(axis=-1) == 0)  # Mask only the feature vectors that are 0 everywhere

            # Calculate reconstruction residuals
            residual_2D = (restored_2D - rgb_patch).pow(2).sum(1).sqrt()
            residual_3D = (restored_3D - xyz_patch).pow(2).sum(1).sqrt()

            # Combine residuals
            residual_comb = (residual_2D * residual_3D)
            residual_comb[xyz_mask] = 0.0

            # Apply Gaussian blur approximation
            residual_comb = residual_comb.reshape(1, 1, 224, 224)
            for _ in range(5):
                residual_comb = torch.nn.functional.conv2d(input=residual_comb, padding=pad_l, weight=weight_l)
            for _ in range(3):
                residual_comb = torch.nn.functional.conv2d(input=residual_comb, padding=pad_u, weight=weight_u)
            residual_comb = residual_comb.reshape(224, 224)

            # Prediction and ground-truth accumulation
            gts.append(gt.squeeze().cpu().detach().numpy())  # (224, 224)
            predictions.append((residual_comb / (residual_comb[residual_comb != 0].mean())).cpu().detach().numpy())  # (224, 224)

            # GTs
            image_labels.append(label)  # (1,)
            pixel_labels.extend(gt.flatten().cpu().detach().numpy())  # (50176,)

            # Predictions
            image_preds.append((residual_comb / torch.sqrt(residual_comb[residual_comb != 0].mean())).cpu().detach().numpy().max())  # (1,)
            pixel_preds.extend((residual_comb / torch.sqrt(residual_comb.mean())).flatten().cpu().detach().numpy())  # (50176,)

            if args.produce_qualitatives:
                defect_class_str = rgb_path[0].split('/')[-3]
                image_name_str = rgb_path[0].split('/')[-1]

                save_path = f'{args.qualitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{defect_class_str}'

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                fig, axs = plt.subplots(2, 3, figsize=(7, 7))

                denormalize = transforms.Compose([
                    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                ])

                rgb = denormalize(rgb)

                os.path.join(save_path, image_name_str)

                axs[0, 0].imshow(rgb.squeeze().permute(1, 2, 0).cpu().detach().numpy())
                axs[0, 0].set_title('RGB')

                axs[0, 1].imshow(gt.squeeze().cpu().detach().numpy())
                axs[0, 1].set_title('Ground-truth')

                axs[0, 2].imshow(depth.squeeze().float().permute(1, 2, 0).mean(axis=-1).cpu().detach().numpy())
                axs[0, 2].set_title('Depth')

                # Reshape tensors for visualization
                residual_3D_reshaped = residual_3D.reshape(224, 224).cpu().detach().numpy()
                residual_2D_reshaped = residual_2D.reshape(224, 224).cpu().detach().numpy()
                residual_comb_reshaped = residual_comb.reshape(224, 224).cpu().detach().numpy()

                axs[1, 0].imshow(residual_3D_reshaped, cmap=plt.cm.jet)
                axs[1, 0].set_title('3D Residual')

                axs[1, 1].imshow(residual_2D_reshaped, cmap=plt.cm.jet)
                axs[1, 1].set_title('2D Residual')

                axs[1, 2].imshow(residual_comb_reshaped, cmap=plt.cm.jet)
                axs[1, 2].set_title('Combined Residual')

                # Remove ticks and labels from all subplots
                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                # Adjust the layout and spacing
                plt.tight_layout()

                plt.savefig(os.path.join(save_path, image_name_str), dpi=256)

                if args.visualize_plot:
                    plt.show()

    # Calculate AD&S metrics
    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    # Save results
    result_file_name = f'{args.quantitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'
    title_string = f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs'
    header_string = 'AUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC & I-AUROC'
    results_string = f'{au_pros[0]:.3f} & {au_pros[1]:.3f} & {au_pros[2]:.3f} & {au_pros[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}'

    if not os.path.exists(args.quantitative_folder):
        os.makedirs(args.quantitative_folder)

    with open(result_file_name, "w") as markdown_file:
        markdown_file.write(title_string + '\n' + header_string + '\n' + results_string)

    # Print AD&S metrics
    print(title_string)
    print("AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC | I-AUROC")
    print(f'  {au_pros[0]:.3f}   |   {au_pros[1]:.3f}   |   {au_pros[2]:.3f}  |   {au_pros[3]:.3f}  |   {pixel_rocauc:.3f} |   {image_rocauc:.3f}', end='\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make inference with Crossmodal Feature Networks (CFMs) on a dataset.')

    parser.add_argument('--dataset_path', default='./datasets/mvtec3d', type=str, help='Dataset path.')
    parser.add_argument('--class_name', default=None, type=str, choices=["chairs","bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
                                                                        'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help='Category name.')
    parser.add_argument('--checkpoint_folder', default='./checkpoints/mvtec', type=str, help='Path to the folder containing CFMs checkpoints.')
    parser.add_argument('--qualitative_folder', default='./results/mvtec-250/qualitatives_mvtec_CBAM', type=str, help='Path to the folder in which to save the qualitatives.')
    parser.add_argument('--quantitative_folder', default='./results/mvtec-250/quantitatives_mvtec_CBAM', type=str, help='Path to the folder in which to save the quantitatives.')
    parser.add_argument('--epochs_no', default=250, type=int, help='Number of epochs to train the CFMs.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch dimension. Usually 16 is around the max.')
    parser.add_argument('--visualize_plot', default=False, action='store_true', help='Whether to show plot or not.')
    parser.add_argument('--produce_qualitatives', default=False, action='store_true', help='Whether to produce qualitatives or not.')
    args = parser.parse_args()

    infer_CFM(args)
