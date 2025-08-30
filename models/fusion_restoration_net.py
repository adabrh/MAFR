import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        x = self.channel_attention(x)
        # Apply spatial attention
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return x * self.sigmoid(out.unsqueeze(-1).unsqueeze(-1))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class FusionEncoder(nn.Module):
    def __init__(self, in_features_2D, in_features_3D, out_features, hidden_dim=None, num_layers=2, dropout=0.1):
        """
        Enhanced FusionEncoder with more flexibility.
       
        Args:
            in_features_2D (int): Input dimension for 2D features.
            in_features_3D (int): Input dimension for 3D features.
            out_features (int): Output dimension.
            hidden_dim (int, optional): Hidden dimension for intermediate layers. Defaults to out_features.
            num_layers (int, optional): Number of fully connected layers. Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(FusionEncoder, self).__init__()
       
        # If hidden_dim is not provided, set it to out_features
        hidden_dim = hidden_dim if hidden_dim is not None else out_features
       
        # Input layer to handle concatenated 2D and 3D features
        self.input_fc = nn.Linear(in_features_2D + in_features_3D, hidden_dim)
       
        # Intermediate fully connected layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
       
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, out_features)
       
        # Activation function
        self.activation = nn.GELU()
       
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
       
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_2D, x_3D):
        """
        Forward pass for the FusionEncoder.
       
        Args:
            x_2D (torch.Tensor): 2D features of shape (batch_size, in_features_2D).
            x_3D (torch.Tensor): 3D features of shape (batch_size, in_features_3D).
           
        Returns:
            torch.Tensor: Output of shape (batch_size, out_features).
        """
        # Concatenate 2D and 3D features along the last dimension
        x = torch.cat((x_2D, x_3D), dim=-1)
       
        # Pass through the input layer
        x = self.activation(self.input_fc(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
       
        # Pass through intermediate layers
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.layer_norm(x)
            x = self.dropout(x)
       
        # Pass through the output layer
        x = self.output_fc(x)
       
        return x
    
class DecoupledDecoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=None, num_layers=3, dropout=0.1):
        super(DecoupledDecoder, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else out_features
        
        # Input layer
        self.input_fc = nn.Linear(in_features, hidden_dim)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, out_features)
        
        # Activation and normalization
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        
        # CBAM
        self.cbam = CBAM(channels=hidden_dim)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip(x)  # Skip connection
        x = self.activation(self.norm(self.input_fc(x)))
        x = self.dropout(x)
        
        # Pass through hidden layers
        for layer in self.layers:
            x = self.activation(self.norm(layer(x)))
            x = self.dropout(x)
        
        # Reshape for CBAM (assuming x is 2D, reshape to 4D)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, hidden_dim, 1, 1)
        x = self.cbam(x)  # Apply CBAM
        x = x.squeeze(-1).squeeze(-1)  # Reshape back to 2D
        
        x = self.output_fc(x) + residual  # Add skip connection
        return x