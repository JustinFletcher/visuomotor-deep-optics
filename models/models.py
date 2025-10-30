#!/usr/bin/env python3
"""
Model architectures for machine learning tasks.
Reusable neural network components that can be combined for different objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def center_crop_transform(tensor, crop_size):
    """
    Center crop a tensor to crop_size × crop_size pixels.
    
    Args:
        tensor: Input tensor of shape [channels, height, width] or [batch, channels, height, width]
        crop_size: Size of the center crop (crop_size × crop_size)
        
    Returns:
        Center-cropped tensor of shape [channels, crop_size, crop_size] or [batch, channels, crop_size, crop_size]
    """
    if len(tensor.shape) == 3:
        # Single image: [C, H, W]
        c, h, w = tensor.shape
        
        # Note: Error checking disabled for TorchScript compatibility
        # Assumes crop_size <= min(h, w)
        
        # Calculate center crop coordinates
        center_h, center_w = h // 2, w // 2
        half_crop = crop_size // 2
        
        start_h = center_h - half_crop
        end_h = start_h + crop_size
        start_w = center_w - half_crop
        end_w = start_w + crop_size
        
        # Perform center crop
        return tensor[:, start_h:end_h, start_w:end_w]
        
    elif len(tensor.shape) == 4:
        # Batch of images: [N, C, H, W]
        n, c, h, w = tensor.shape
        
        # Note: Error checking disabled for TorchScript compatibility
        # Assumes crop_size <= min(h, w)
        
        # Calculate center crop coordinates
        center_h, center_w = h // 2, w // 2
        half_crop = crop_size // 2
        
        start_h = center_h - half_crop
        end_h = start_h + crop_size
        start_w = center_w - half_crop
        end_w = start_w + crop_size
        
        # Perform center crop
        return tensor[:, :, start_h:end_h, start_w:end_w]
    
    else:
        raise ValueError(f"Expected 3D tensor [C, H, W] or 4D tensor [N, C, H, W], got shape {tensor.shape}")


class VanillaCNN(nn.Module):
    """CNN model for predicting actions from observations"""
    
    def __init__(self, input_channels=2, action_dim=15, input_crop_size=None):
        """
        Args:
            input_channels: Number of observation channels (2 for real/imag)
            action_dim: Dimension of action space (15 for optomech segments)
            input_crop_size: If specified, center crop input to this size (e.g., 128)
        """
        super(VanillaCNN, self).__init__()
        self.input_crop_size = input_crop_size
        
        # CNN feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class BasicBlockGroupNorm(nn.Module):
    """Basic ResNet block with GroupNorm"""
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(32, planes//4), num_channels=planes)  # Better group sizing
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(32, planes//4), num_channels=planes)  # Better group sizing
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(32, planes//4), num_channels=planes)  # Better group sizing
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        # Add residual connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out



class ResNet18GroupNorm(nn.Module):
    def __init__(self, input_channels=2, action_dim=15, input_crop_size=None):
        super().__init__()
        self.input_crop_size = input_crop_size
        # Input group default
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=3, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=32)  # Use 16 groups for 32 channels
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Input group default
        # self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)  # Use 8 groups for 64 channels
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # ResNet18 layer configuration: [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=1)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)  # Added missing layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)  # Changed from 256 to 128
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(128, action_dim)  # Changed from 256 to 128
        self.tanh = nn.Tanh()

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        layers = []
        
        # First block (may have downsample)
        layers.append(BasicBlockGroupNorm(in_planes, planes, stride))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(BasicBlockGroupNorm(planes, planes, stride=1))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Apply center cropping if specified
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Added layer4 forward pass
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x


class AutoEncoderCNN(nn.Module):
    """Convolutional Autoencoder with separable encoder and decoder components"""
    
    def __init__(self, input_channels=2, latent_dim=256, input_size=256, input_crop_size=None):
        """
        Args:
            input_channels: Number of input channels (2 for real/imag)
            latent_dim: Dimension of the latent representation
            input_size: Expected input size after any cropping (default 256)
            input_crop_size: If specified, center crop input to this size before processing
        """
        super(AutoEncoderCNN, self).__init__()
        self.input_crop_size = input_crop_size
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Calculate number of downsampling layers to reach 4x4 feature map
        self.num_downsample = self._calculate_downsample_layers(input_size)
        
        # Encoder: reduces spatial dimensions and increases channels
        encoder_layers = []
        current_channels = input_channels
        
        # Build encoder layers dynamically
        channel_progression = [32, 64, 128, 256, 512]
        for i in range(self.num_downsample):
            out_channels = channel_progression[min(i, len(channel_progression) - 1)]
            encoder_layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_channels
        
        # Add adaptive pooling to ensure 4x4 output
        encoder_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.final_channels = current_channels
        
        # Bottleneck: compress to latent representation
        self.bottleneck_encode = nn.Sequential(
            nn.Linear(self.final_channels * 4 * 4, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck: expand from latent representation
        self.bottleneck_decode = nn.Sequential(
            nn.Linear(latent_dim, self.final_channels * 4 * 4),
            nn.ReLU(inplace=True)
        )
        
        # Build decoder layers dynamically to reconstruct to input_size
        decoder_layers = []
        current_channels = self.final_channels
        current_size = 4
        
        # Reverse channel progression
        channel_progression = [256, 128, 64, 32, 16]
        
        # Build decoder layers to upsample back to input_size
        layer_count = 0
        while current_size < input_size:
            if layer_count < len(channel_progression):
                out_channels = channel_progression[layer_count]
            else:
                out_channels = 16  # Keep final channels low
            
            decoder_layers.extend([
                nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_channels
            current_size *= 2
            layer_count += 1
        
        # Final layer to output channels
        decoder_layers.extend([
            nn.ConvTranspose2d(current_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _calculate_downsample_layers(self, input_size):
        """Calculate number of downsampling layers needed to reach 4x4"""
        size = input_size
        layers = 0
        while size > 4:
            size //= 2
            layers += 1
        return max(layers, 1)  # At least one layer
        
    def encode(self, x):
        """Encode input to latent representation"""
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.bottleneck_encode(x)
        return x
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        x = self.bottleneck_decode(z)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape for decoder
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoderResNet(nn.Module):
    """ResNet-based Autoencoder with separable encoder and decoder components"""
    
    def __init__(self, input_channels=2, latent_dim=512, input_size=256, input_crop_size=None):
        """
        Args:
            input_channels: Number of input channels (2 for real/imag) 
            latent_dim: Dimension of the latent representation
            input_size: Expected input size after any cropping (default 256)
            input_crop_size: If specified, center crop input to this size before processing
        """
        super(AutoEncoderResNet, self).__init__()
        self.input_crop_size = input_crop_size
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder: ResNet-style encoder
        self.encoder = nn.Sequential(
            # Initial conv
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet blocks
            self._make_encoder_layer(64, 64, 2, stride=1),
            self._make_encoder_layer(64, 128, 2, stride=2),
            self._make_encoder_layer(128, 256, 2, stride=2),
            self._make_encoder_layer(256, 512, 2, stride=2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Bottleneck
        self.bottleneck_encode = nn.Sequential(
            nn.Linear(512 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck_decode = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: transposed convolutions
        self.decoder = nn.Sequential(
            # Upsample from 4x4 to target size
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1), # 256x256
            nn.Sigmoid()
        )
        
    def _make_encoder_layer(self, in_planes, planes, blocks, stride=1):
        """Create a ResNet encoder layer"""
        layers = []
        # First block (may have downsample)
        layers.append(BasicBlockGroupNorm(in_planes, planes, stride))
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(BasicBlockGroupNorm(planes, planes, stride=1))
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """Encode input to latent representation"""
        if self.input_crop_size is not None:
            x = center_crop_transform(x, self.input_crop_size)
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.bottleneck_encode(x)
        return x
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        x = self.bottleneck_decode(z)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape for decoder
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ResNet18Actor(nn.Module):
    """
    ResNet-18 based actor for behavior cloning and imitation learning.
    
    Supports loading pre-trained encoders from autoencoder models for transfer learning.
    The encoder can be frozen or fine-tuned during training.
    """
    
    def __init__(self, input_channels: int = 2, action_dim: int = 4, 
                 pretrained_encoder_path: str = None, freeze_encoder: bool = False):
        """
        Args:
            input_channels: Number of input channels (2 for complex image data)
            action_dim: Dimensionality of action space
            pretrained_encoder_path: Path to saved autoencoder model to load encoder from
            freeze_encoder: If True, freeze encoder weights during training
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.action_dim = action_dim
        self.freeze_encoder = freeze_encoder
        
        # Try to load pre-trained encoder if path provided
        if pretrained_encoder_path is not None:
            print(f"🔄 Loading pre-trained encoder from {pretrained_encoder_path}")
            self._load_pretrained_encoder(pretrained_encoder_path)
        else:
            # Create ResNet-18 backbone from scratch
            import torchvision.models as models
            self.resnet = models.resnet18(weights=None)
            
            # Modify first conv layer to accept input_channels
            if input_channels != 3:
                self.resnet.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            
            # Get feature dimension from ResNet
            self.feature_dim = self.resnet.fc.in_features
            
            # Remove the final FC layer (we'll add our own action head)
            self.resnet.fc = nn.Identity()
        
        # Create action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )
        
        # Freeze encoder if requested
        if self.freeze_encoder and pretrained_encoder_path is not None:
            self._freeze_encoder()
    
    def _load_pretrained_encoder(self, encoder_path: str):
        """
        Load encoder from a saved autoencoder model.
        
        Supports both AutoEncoderCNN and AutoEncoderResNet architectures.
        """
        import torch
        from pathlib import Path
        import sys
        
        encoder_path = Path(encoder_path)
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        # Load checkpoint - try full load first, fall back to injecting dummy classes if needed
        try:
            checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
        except AttributeError as e:
            # Handle case where checkpoint has classes we don't have in scope
            # Inject a dummy AutoencoderConfig into __main__ module
            print(f"⚠️  Warning: Checkpoint contains classes not in scope")
            print(f"   Injecting dummy AutoencoderConfig class into __main__...")
            
            # Create a dummy class that accepts any arguments
            class AutoencoderConfig:
                def __init__(self, *args, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            # Inject it into __main__ so pickle can find it
            import __main__
            __main__.AutoencoderConfig = AutoencoderConfig
            
            # Try loading again
            checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Try to infer encoder architecture from state dict keys
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
        
        if not encoder_keys:
            print("⚠️  No encoder keys found in checkpoint, trying to load full model...")
            # Might be a full autoencoder - try to extract encoder part
            encoder_keys = [k for k in state_dict.keys() if 'encoder' in k.lower()]
        
        if encoder_keys:
            # Create a temporary autoencoder to load the state dict
            # We'll extract just the encoder part
            from models.models import AutoEncoderResNet, AutoEncoderCNN
            
            # Infer latent_dim from the checkpoint by looking at the bottleneck layer size
            latent_dim = None
            if 'bottleneck_encode.0.weight' in state_dict:
                latent_dim = state_dict['bottleneck_encode.0.weight'].shape[0]
                print(f"📐 Inferred latent_dim={latent_dim} from checkpoint")
            
            # Infer input_channels from encoder first layer
            input_channels = self.input_channels
            if 'encoder.0.weight' in state_dict:
                input_channels = state_dict['encoder.0.weight'].shape[1]
                print(f"📐 Inferred input_channels={input_channels} from checkpoint")
            
            # Try AutoEncoderResNet first (more common)
            try:
                temp_model = AutoEncoderResNet(
                    input_channels=input_channels, 
                    latent_dim=latent_dim if latent_dim else 512,
                    input_size=256
                )
                temp_model.load_state_dict(state_dict, strict=False)
                
                # Extract encoder plus bottleneck as our backbone  
                # We need to include the bottleneck to get the correct feature dimension
                class EncoderWithBottleneck(nn.Module):
                    def __init__(self, encoder, bottleneck):
                        super().__init__()
                        self.encoder = encoder
                        self.bottleneck = bottleneck
                    
                    def forward(self, x):
                        x = self.encoder(x)
                        x = x.view(x.size(0), -1)  # Flatten
                        x = self.bottleneck(x)
                        return x
                
                self.resnet = EncoderWithBottleneck(temp_model.encoder, temp_model.bottleneck_encode)
                self.feature_dim = temp_model.latent_dim
                
                print(f"✅ Loaded ResNet encoder with latent_dim={self.feature_dim}")
                
            except Exception as e:
                print(f"⚠️  Could not load as AutoEncoderResNet: {e}")
                try:
                    # Try AutoEncoderCNN
                    temp_model = AutoEncoderCNN(
                        input_channels=input_channels,
                        latent_dim=latent_dim if latent_dim else 512,
                        input_size=256
                    )
                    temp_model.load_state_dict(state_dict, strict=False)
                    
                    # Extract encoder plus bottleneck
                    class EncoderWithBottleneck(nn.Module):
                        def __init__(self, encoder, bottleneck):
                            super().__init__()
                            self.encoder = encoder
                            self.bottleneck = bottleneck
                        
                        def forward(self, x):
                            x = self.encoder(x)
                            x = x.view(x.size(0), -1)  # Flatten
                            x = self.bottleneck(x)
                            return x
                    
                    self.resnet = EncoderWithBottleneck(temp_model.encoder, temp_model.bottleneck_encode)
                    self.feature_dim = temp_model.latent_dim
                    
                    print(f"✅ Loaded CNN encoder with latent_dim={self.feature_dim}")
                    
                except Exception as e2:
                    print(f"❌ Could not load encoder: {e2}")
                    raise ValueError(f"Failed to load encoder from {encoder_path}")
        else:
            # No encoder found, try loading as a standalone ResNet
            import torchvision.models as models
            self.resnet = models.resnet18(weights=None)
            
            if self.input_channels != 3:
                self.resnet.conv1 = nn.Conv2d(
                    self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            
            self.feature_dim = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
            
            # Try to load weights
            self.resnet.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded ResNet weights with feature_dim={self.feature_dim}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device='cpu'):
        """
        Create a ResNet18Actor from a checkpoint, automatically detecting the architecture.
        
        This handles both models trained from scratch and models using pre-trained encoders.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            
        Returns:
            ResNet18Actor instance with loaded weights
        """
        import torch
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Detect if this model uses a pre-trained encoder architecture
        has_pretrained_encoder = 'resnet.encoder.0.weight' in state_dict
        
        # Extract input_channels
        if has_pretrained_encoder:
            input_channels = state_dict['resnet.encoder.0.weight'].shape[1]
        elif 'resnet.conv1.weight' in state_dict:
            input_channels = state_dict['resnet.conv1.weight'].shape[1]
        else:
            raise ValueError("Could not detect input_channels from checkpoint")
        
        # Extract action_dim
        if 'action_head.3.weight' in state_dict:
            action_dim = state_dict['action_head.3.weight'].shape[0]
        elif 'action_head.0.weight' in state_dict:
            action_dim = state_dict['action_head.0.weight'].shape[0]
        else:
            raise ValueError("Could not detect action_dim from checkpoint")
        
        print(f"🔍 Detected from checkpoint: input_channels={input_channels}, action_dim={action_dim}, has_pretrained_encoder={has_pretrained_encoder}")
        
        # Create model - if it has pre-trained encoder architecture, we need to reconstruct it
        if has_pretrained_encoder:
            # Get latent_dim from encoder bottleneck
            if 'resnet.bottleneck.0.weight' in state_dict:
                latent_dim = state_dict['resnet.bottleneck.0.weight'].shape[0]
            else:
                raise ValueError("Could not detect latent_dim from pre-trained encoder checkpoint")
            
            print(f"🔍 Detected latent_dim={latent_dim} from encoder bottleneck")
            
            # Reconstruct the encoder architecture by creating a dummy autoencoder
            temp_autoencoder = AutoEncoderResNet(
                input_channels=input_channels,
                latent_dim=latent_dim,
                input_size=256
            )
            
            # Extract just the encoder and bottleneck parts from state_dict
            encoder_state = {k.replace('resnet.', ''): v for k, v in state_dict.items() if k.startswith('resnet.encoder.') or k.startswith('resnet.bottleneck.')}
            temp_autoencoder.load_state_dict(encoder_state, strict=False)
            
            # Create wrapper for encoder + bottleneck
            class EncoderWithBottleneck(nn.Module):
                def __init__(self, encoder, bottleneck):
                    super().__init__()
                    self.encoder = encoder
                    self.bottleneck = bottleneck
                
                def forward(self, x):
                    x = self.encoder(x)
                    x = x.view(x.size(0), -1)
                    x = self.bottleneck(x)
                    return x
            
            # Create model with the reconstructed encoder
            model = object.__new__(cls)
            nn.Module.__init__(model)  # Call parent __init__
            
            model.input_channels = input_channels
            model.action_dim = action_dim
            model.freeze_encoder = False
            model.feature_dim = latent_dim
            model.resnet = EncoderWithBottleneck(temp_autoencoder.encoder, temp_autoencoder.bottleneck_encode)
            
            # Recreate action head with correct input dimension
            model.action_head = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, action_dim)
            )
            
            # Now load the action_head weights
            action_head_state = {k.replace('action_head.', ''): v for k, v in state_dict.items() if k.startswith('action_head.')}
            model.action_head.load_state_dict(action_head_state, strict=True)
            
            print(f"✅ Reconstructed model with pre-trained encoder architecture (latent_dim={latent_dim})")
        else:
            # Standard ResNet18 architecture - create and load normally
            model = cls(input_channels=input_channels, action_dim=action_dim, pretrained_encoder_path=None)
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ Loaded standard ResNet18Actor from checkpoint")
        
        return model
    
    def _freeze_encoder(self):
        """Freeze encoder weights for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        print("🔒 Encoder weights frozen for fine-tuning")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        self.freeze_encoder = False
        print("🔓 Encoder weights unfrozen")
    
    def forward(self, x):
        """
        Forward pass through encoder and action head.
        
        Args:
            x: Input observations [batch, channels, height, width]
            
        Returns:
            Predicted actions [batch, action_dim]
        """
        # Extract features with encoder
        features = self.resnet(x)
        
        # Flatten if needed (for CNN encoders)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Predict actions
        actions = self.action_head(features)
        
        return actions


def create_model(arch: str, input_channels: int, action_dim: int = None, channel_scale: int = 16, mlp_scale: int = 128, input_crop_size: int = None, latent_dim: int = 256, input_size: int = 256, pretrained_encoder_path: str = None, freeze_encoder: bool = False) -> nn.Module:
    """Factory function to create different model architectures"""
    if arch == "vanilla_cnn":
        if action_dim is None:
            raise ValueError("action_dim required for vanilla_cnn")
        return VanillaCNN(input_channels=input_channels, action_dim=action_dim, input_crop_size=input_crop_size)
    elif arch == "resnet18_gn":
        if action_dim is None:
            raise ValueError("action_dim required for resnet18_gn")
        return ResNet18GroupNorm(input_channels=input_channels, action_dim=action_dim, input_crop_size=input_crop_size)
    elif arch == "resnet18_actor":
        if action_dim is None:
            raise ValueError("action_dim required for resnet18_actor")
        return ResNet18Actor(input_channels=input_channels, action_dim=action_dim, 
                            pretrained_encoder_path=pretrained_encoder_path, 
                            freeze_encoder=freeze_encoder)
    elif arch == "autoencoder_cnn":
        return AutoEncoderCNN(input_channels=input_channels, latent_dim=latent_dim, input_size=input_size, input_crop_size=input_crop_size)
    elif arch == "autoencoder_resnet":
        return AutoEncoderResNet(input_channels=input_channels, latent_dim=latent_dim, input_size=input_size, input_crop_size=input_crop_size)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
