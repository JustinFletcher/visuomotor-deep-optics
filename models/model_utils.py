#!/usr/bin/env python3
"""
Model utilities for saving, loading, and managing trained models.
Provides codebase-independent model persistence and component extraction.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pickle
import importlib.util
import sys
from datetime import datetime

from .models import create_model, VanillaCNN, ResNet18GroupNorm


class ModelRegistry:
    """Registry for managing trained models and their metadata"""
    
    def __init__(self, registry_dir: str = "saved_models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "model_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load the model registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def _save_registry(self):
        """Save the model registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_id: str, metadata: Dict[str, Any]):
        """Register a new model with metadata"""
        self.registry[model_id] = {
            **metadata,
            'registered_at': datetime.now().isoformat(),
            'registry_version': '1.0'
        }
        self._save_registry()
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered model"""
        return self.registry.get(model_id)
    
    def list_models(self, architecture: str = None, task: str = None) -> List[str]:
        """List registered models, optionally filtered by architecture or task"""
        models = []
        for model_id, metadata in self.registry.items():
            if architecture and metadata.get('architecture') != architecture:
                continue
            if task and metadata.get('task') != task:
                continue
            models.append(model_id)
        return models
    
    def remove_model(self, model_id: str):
        """Remove a model from the registry"""
        if model_id in self.registry:
            del self.registry[model_id]
            self._save_registry()
            
            # Also remove the model file if it exists
            model_path = self.registry_dir / f"{model_id}.pth"
            if model_path.exists():
                model_path.unlink()


class ModelSaver:
    """Utility for saving models with codebase independence"""
    
    @staticmethod
    def save_model(model: nn.Module, 
                   model_id: str,
                   metadata: Dict[str, Any],
                   save_dir: str = "saved_models",
                   save_components: bool = True,
                   save_torchscript: bool = False,
                   example_input: Optional[torch.Tensor] = None) -> str:
        """
        Save a model with full metadata and optional component extraction
        
        Args:
            model: The trained PyTorch model
            model_id: Unique identifier for the model
            metadata: Model metadata (architecture, config, training info, etc.)
            save_dir: Directory to save models
            save_components: Whether to save extractable components
            save_torchscript: Whether to save TorchScript version
            example_input: Example input for TorchScript tracing
            
        Returns:
            Path to the saved model file
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        model_file = save_path / f"{model_id}.pth"
        
        # Prepare the save dictionary
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class_name': model.__class__.__name__,
            'model_module': model.__class__.__module__,
            'metadata': metadata,
            'save_timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        # Save extractable components if requested
        if save_components:
            components = ModelSaver._extract_components(model)
            save_dict['components'] = components
        
        # Save the main model
        torch.save(save_dict, model_file)
        
        # Save TorchScript version if requested
        if save_torchscript and example_input is not None:
            try:
                model.eval()
                traced_model = torch.jit.trace(model, example_input)
                torchscript_file = save_path / f"{model_id}_traced.pt"
                traced_model.save(str(torchscript_file))
                save_dict['torchscript_available'] = True
            except Exception as e:
                print(f"Warning: Could not save TorchScript version: {e}")
                save_dict['torchscript_available'] = False
        
        # Register the model
        registry = ModelRegistry(save_dir)
        registry.register_model(model_id, {
            **metadata,
            'model_file': str(model_file),
            'components_available': save_components,
            'torchscript_available': save_dict.get('torchscript_available', False)
        })
        
        return str(model_file)
    
    @staticmethod
    def _extract_components(model: nn.Module) -> Dict[str, Any]:
        """Extract reusable components from a model"""
        components = {}
        
        # Extract common components based on model type
        if hasattr(model, 'features'):
            components['features'] = {
                'state_dict': model.features.state_dict(),
                'type': 'feature_extractor'
            }
        
        if hasattr(model, 'classifier'):
            components['classifier'] = {
                'state_dict': model.classifier.state_dict(),
                'type': 'classifier'
            }
        
        if hasattr(model, 'conv1'):
            components['conv1'] = {
                'state_dict': model.conv1.state_dict(),
                'type': 'initial_conv'
            }
        
        # Extract ResNet-style layers
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                components[layer_name] = {
                    'state_dict': layer.state_dict(),
                    'type': 'resnet_layer'
                }
        
        return components


class ModelLoader:
    """Utility for loading models with codebase independence"""
    
    @staticmethod
    def load_model(model_path: str, 
                   device: str = 'cpu',
                   strict: bool = True) -> tuple[nn.Module, Dict[str, Any]]:
        """
        Load a model from disk with full metadata
        
        Args:
            model_path: Path to the saved model file
            device: Device to load the model on
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract metadata
        metadata = checkpoint.get('metadata', {})
        
        # Recreate the model using the factory function
        architecture = metadata.get('architecture')
        model_config = metadata.get('model_config', {})
        
        if architecture:
            model = create_model(architecture, **model_config)
        else:
            # Fallback: try to recreate using class information
            model_class_name = checkpoint.get('model_class_name')
            if model_class_name == 'VanillaCNN':
                model = VanillaCNN(**model_config)
            elif model_class_name == 'ResNet18GroupNorm':
                model = ResNet18GroupNorm(**model_config)
            else:
                raise ValueError(f"Cannot recreate model of type {model_class_name}")
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        return model, metadata
    
    @staticmethod
    def load_component(model_path: str, 
                       component_name: str,
                       target_module: nn.Module,
                       device: str = 'cpu',
                       strict: bool = True) -> nn.Module:
        """
        Load a specific component from a saved model into a target module
        
        Args:
            model_path: Path to the saved model file
            component_name: Name of the component to extract
            target_module: Module to load the component into
            device: Device to load on
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            The target module with loaded weights
        """
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'components' not in checkpoint:
            raise ValueError(f"No components available in {model_path}")
        
        if component_name not in checkpoint['components']:
            available = list(checkpoint['components'].keys())
            raise ValueError(f"Component '{component_name}' not found. Available: {available}")
        
        component_data = checkpoint['components'][component_name]
        target_module.load_state_dict(component_data['state_dict'], strict=strict)
        
        return target_module
    
    @staticmethod
    def load_torchscript(model_path: str, device: str = 'cpu') -> torch.jit.ScriptModule:
        """Load a TorchScript model"""
        if not model_path.endswith('_traced.pt'):
            model_path = model_path.replace('.pth', '_traced.pt')
        
        return torch.jit.load(model_path, map_location=device)


class ComponentExtractor:
    """Utility for creating new models using components from trained models"""
    
    @staticmethod
    def create_hybrid_model(base_architecture: str,
                           pretrained_components: Dict[str, str],
                           model_config: Dict[str, Any],
                           device: str = 'cpu') -> nn.Module:
        """
        Create a new model using components from different trained models
        
        Args:
            base_architecture: Base architecture to start with
            pretrained_components: Dict of {component_name: model_path}
            model_config: Configuration for the base model
            device: Device to create the model on
            
        Returns:
            New model with pretrained components loaded
        """
        # Create the base model
        model = create_model(base_architecture, **model_config)
        model.to(device)
        
        # Load pretrained components
        for component_name, model_path in pretrained_components.items():
            if hasattr(model, component_name):
                target_module = getattr(model, component_name)
                ModelLoader.load_component(
                    model_path, component_name, target_module, device
                )
                print(f"Loaded pretrained {component_name} from {model_path}")
            else:
                print(f"Warning: Model does not have component '{component_name}'")
        
        return model
    
    @staticmethod
    def freeze_components(model: nn.Module, component_names: List[str]):
        """Freeze specific components of a model"""
        for component_name in component_names:
            if hasattr(model, component_name):
                component = getattr(model, component_name)
                for param in component.parameters():
                    param.requires_grad = False
                print(f"Frozen component: {component_name}")
            else:
                print(f"Warning: Component '{component_name}' not found")


# Convenience functions for easy usage
def save_trained_model(model: nn.Module,
                      model_id: str,
                      architecture: str,
                      model_config: Dict[str, Any],
                      training_info: Dict[str, Any] = None,
                      task: str = None,
                      save_dir: str = "saved_models",
                      example_input: torch.Tensor = None) -> str:
    """Convenience function to save a trained model with all metadata"""
    
    metadata = {
        'architecture': architecture,
        'model_config': model_config,
        'task': task or 'unknown',
        'training_info': training_info or {}
    }
    
    return ModelSaver.save_model(
        model=model,
        model_id=model_id,
        metadata=metadata,
        save_dir=save_dir,
        save_components=True,
        save_torchscript=example_input is not None,
        example_input=example_input
    )


def load_trained_model(model_id: str,
                      save_dir: str = "saved_models",
                      device: str = 'cpu') -> tuple[nn.Module, Dict[str, Any]]:
    """Convenience function to load a trained model by ID"""
    
    registry = ModelRegistry(save_dir)
    model_info = registry.get_model_info(model_id)
    
    if model_info is None:
        raise ValueError(f"Model '{model_id}' not found in registry")
    
    model_path = model_info['model_file']
    return ModelLoader.load_model(model_path, device)


def create_model_with_pretrained_features(architecture: str,
                                         feature_extractor_model_id: str,
                                         model_config: Dict[str, Any],
                                         save_dir: str = "saved_models",
                                         device: str = 'cpu') -> nn.Module:
    """Create a model with pretrained feature extractor"""
    
    registry = ModelRegistry(save_dir)
    feature_model_info = registry.get_model_info(feature_extractor_model_id)
    
    if feature_model_info is None:
        raise ValueError(f"Feature extractor model '{feature_extractor_model_id}' not found")
    
    return ComponentExtractor.create_hybrid_model(
        base_architecture=architecture,
        pretrained_components={'features': feature_model_info['model_file']},
        model_config=model_config,
        device=device
    )
