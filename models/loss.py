import torch
import torch.nn.functional as F
import numpy as np

def image_similarity_loss(I1, I1_warped, epsilon=1e-8):
    """
    Zero-Normalized Sum of Squared Differences (ZNSSD) loss for 2D tensors.
    Args:
        I1: Reference image (height * width, num_features).
        I1_warped: Warped image (height * width, num_features).
        epsilon: Small constant to avoid division by zero.
    Returns:
        ZNSSD loss.
    """
    # Compute mean and standard deviation along the feature dimension
    I1_mean = I1.mean(dim=1, keepdim=True)  # Shape: (height * width, 1)
    I1_warped_mean = I1_warped.mean(dim=1, keepdim=True)  # Shape: (height * width, 1)
    
    I1_std = torch.sqrt(((I1 - I1_mean) ** 2).mean(dim=1, keepdim=True) + epsilon)  # Shape: (height * width, 1)
    I1_warped_std = torch.sqrt(((I1_warped - I1_warped_mean) ** 2).mean(dim=1, keepdim=True) + epsilon)  # Shape: (height * width, 1)
    
    # Normalize images
    I1_normalized = (I1 - I1_mean) / I1_std  # Shape: (height * width, num_features)
    I1_warped_normalized = (I1_warped - I1_warped_mean) / I1_warped_std  # Shape: (height * width, num_features)
    
    # Compute ZNSSD
    loss = torch.mean((I1_normalized - I1_warped_normalized) ** 2)  # Scalar
    return loss

def smoothness_loss(deformation_field, I1, epsilon=1e-8):
    """
    Smoothness loss for 2D tensors with multi-channel features.
    Args:
        deformation_field: Estimated deformation field (height * width, 2).
        I1: Reference image (height * width, num_features).
        epsilon: Small constant to avoid division by zero.
    Returns:
        Smoothness loss.
    """
    # Reshape to (batch, channels, height, width)
    height = int(np.sqrt(I1.shape[0]))  # Assuming height = width
    I1_reshaped = I1.view(height, height, -1).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, num_features, height, width)
    deformation_field_reshaped = deformation_field.view(height, height, -1).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 2, height, width)
    
    # Reduce I1 to a single channel by taking the mean across feature dimension
    I1_reduced = I1_reshaped.mean(dim=1, keepdim=True)  # Shape: (1, 1, height, width)
    
    # Define convolution kernels for gradient computation
    kernel_x = torch.tensor([[[[-1, 1]]]], device=I1.device).float()  # Shape: (1, 1, 1, 2)
    kernel_y = torch.tensor([[[[-1], [1]]]], device=I1.device).float()  # Shape: (1, 1, 2, 1)
    
    # Compute gradients without padding (output size will be 223x223)
    grad_I1_x = torch.abs(F.conv2d(I1_reduced, kernel_x))  # Shape: (1, 1, 224, 223)
    grad_I1_y = torch.abs(F.conv2d(I1_reduced, kernel_y))  # Shape: (1, 1, 223, 224)
    
    grad_u_x = torch.abs(F.conv2d(deformation_field_reshaped[:, 0:1, :, :], kernel_x))  # Shape: (1, 1, 224, 223)
    grad_u_y = torch.abs(F.conv2d(deformation_field_reshaped[:, 0:1, :, :], kernel_y))  # Shape: (1, 1, 223, 224)
    grad_v_x = torch.abs(F.conv2d(deformation_field_reshaped[:, 1:2, :, :], kernel_x))  # Shape: (1, 1, 224, 223)
    grad_v_y = torch.abs(F.conv2d(deformation_field_reshaped[:, 1:2, :, :], kernel_y))  # Shape: (1, 1, 223, 224)
    
    # Align tensor sizes by cropping to the smallest common dimensions (223x223)
    min_height = min(grad_I1_x.shape[2], grad_I1_y.shape[2])  # 223
    min_width = min(grad_I1_x.shape[3], grad_I1_y.shape[3])   # 223
    
    grad_I1_x = grad_I1_x[:, :, :min_height, :min_width]  # Shape: (1, 1, 223, 223)
    grad_I1_y = grad_I1_y[:, :, :min_height, :min_width]  # Shape: (1, 1, 223, 223)
    grad_u_x = grad_u_x[:, :, :min_height, :min_width]    # Shape: (1, 1, 223, 223)
    grad_u_y = grad_u_y[:, :, :min_height, :min_width]    # Shape: (1, 1, 223, 223)
    grad_v_x = grad_v_x[:, :, :min_height, :min_width]    # Shape: (1, 1, 223, 223)
    grad_v_y = grad_v_y[:, :, :min_height, :min_width]    # Shape: (1, 1, 223, 223)
    
    # Compute smoothness loss
    loss = torch.mean(grad_u_x * torch.exp(-grad_I1_x) + grad_u_y * torch.exp(-grad_I1_y) +
                      grad_v_x * torch.exp(-grad_I1_x) + grad_v_y * torch.exp(-grad_I1_y))
    return loss

def census_loss(I1, I1_warped, patch_size=7):
    """
    Census loss for 2D tensors.
    Args:
        I1: Reference image (height * width, num_features).
        I1_warped: Warped image (height * width, num_features).
        patch_size: Size of the patch for census transform.
    Returns:
        Census loss.
    """
    # Reshape to (height, width, num_features)
    height = int(np.sqrt(I1.shape[0]))  # Assuming height = width
    I1_reshaped = I1.view(height, height, -1).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, num_features, height, width)
    I1_warped_reshaped = I1_warped.view(height, height, -1).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, num_features, height, width)
    
    # Compute census transform using average pooling
    I1_census = F.avg_pool2d(I1_reshaped, kernel_size=patch_size, stride=1, padding=patch_size // 2)
    I1_warped_census = F.avg_pool2d(I1_warped_reshaped, kernel_size=patch_size, stride=1, padding=patch_size // 2)
    
    # Compute census loss
    loss = torch.mean(torch.abs(I1_census - I1_warped_census))
    return loss