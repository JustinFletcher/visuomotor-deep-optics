#!/usr/bin/env python3
"""
Example script demonstrating model utilities for codebase-independent model persistence
and component reuse.

This script shows:
1. How to save models with full metadata
2. How to load models independently of the original codebase
3. How to extract and reuse components from trained models
4. How to create hybrid models with pretrained components
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import our utilities
from models.model_utils import (
    save_trained_model, load_trained_model, create_model_with_pretrained_features,
    ModelRegistry, ModelLoader, ComponentExtractor
)
from models.models import create_model


def demonstrate_model_saving():
    """Demonstrate saving a model with full metadata"""
    print("=" * 60)
    print("🏗️  DEMONSTRATING MODEL SAVING")
    print("=" * 60)
    
    # Create a sample model
    model = create_model(
        arch="vanilla_cnn",
        input_channels=2,
        action_dim=15,
        input_crop_size=128
    )
    
    # Simulate some training (just random weights)
    print("🎯 Training a sample model...")
    
    # Create example input for TorchScript
    example_input = torch.randn(1, 2, 256, 256)
    
    # Model configuration
    model_config = {
        'input_channels': 2,
        'action_dim': 15,
        'input_crop_size': 128
    }
    
    # Training information
    training_info = {
        'dataset_path': 'datasets/example_dataset',
        'target_type': 'sa_incremental_action',
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'best_val_loss': 0.025,
        'final_test_loss': 0.028,
        'optimizer': 'adam'
    }
    
    # Save the model
    model_id = "demo_vanilla_cnn_001"
    save_path = save_trained_model(
        model=model,
        model_id=model_id,
        architecture="vanilla_cnn",
        model_config=model_config,
        training_info=training_info,
        task="behavior_cloning",
        save_dir="saved_models",
        example_input=example_input
    )
    
    print(f"✅ Model saved successfully!")
    print(f"   Model ID: {model_id}")
    print(f"   Path: {save_path}")
    
    return model_id


def demonstrate_model_loading(model_id: str):
    """Demonstrate loading a model independently"""
    print("\n" + "=" * 60)
    print("📦 DEMONSTRATING MODEL LOADING")
    print("=" * 60)
    
    # Load the model
    print(f"🔄 Loading model: {model_id}")
    loaded_model, metadata = load_trained_model(model_id, save_dir="saved_models")
    
    print("✅ Model loaded successfully!")
    print(f"   Architecture: {metadata.get('architecture')}")
    print(f"   Task: {metadata.get('task')}")
    print(f"   Training info: {metadata.get('training_info', {}).get('best_val_loss')}")
    
    # Test the model
    test_input = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        output = loaded_model(test_input)
    print(f"   Test output shape: {output.shape}")
    
    return loaded_model, metadata


def demonstrate_component_extraction(model_id: str):
    """Demonstrate extracting components from a trained model"""
    print("\n" + "=" * 60)
    print("🧩 DEMONSTRATING COMPONENT EXTRACTION")
    print("=" * 60)
    
    # Create a new model with different output size
    print("🏗️  Creating new model with different action dimension...")
    new_model = create_model(
        arch="vanilla_cnn",
        input_channels=2,
        action_dim=30,  # Different action dimension
        input_crop_size=128
    )
    
    # Load pretrained features from the saved model
    print(f"🔄 Loading pretrained features from {model_id}...")
    registry = ModelRegistry("saved_models")
    model_info = registry.get_model_info(model_id)
    
    if model_info and model_info.get('components_available'):
        model_path = model_info['model_file']
        
        # Load the pretrained features
        ModelLoader.load_component(
            model_path=model_path,
            component_name='features',
            target_module=new_model.features,
            strict=False  # Allow size mismatches
        )
        
        print("✅ Pretrained features loaded successfully!")
        
        # Freeze the pretrained features
        ComponentExtractor.freeze_components(new_model, ['features'])
        print("🔒 Pretrained features frozen for fine-tuning")
        
        # Test the hybrid model
        test_input = torch.randn(1, 2, 256, 256)
        with torch.no_grad():
            output = new_model(test_input)
        print(f"   New model output shape: {output.shape}")
        
        return new_model
    else:
        print("❌ No components available for extraction")
        return None


def demonstrate_hybrid_model_creation():
    """Demonstrate creating hybrid models with pretrained components"""
    print("\n" + "=" * 60)
    print("🔀 DEMONSTRATING HYBRID MODEL CREATION")
    print("=" * 60)
    
    # Train a feature extractor model
    print("🎯 Training a feature extractor model...")
    feature_model = create_model(
        arch="vanilla_cnn",  # Use vanilla_cnn which has 'features' component
        input_channels=2,
        action_dim=15,
        input_crop_size=128
    )
    
    # Save the feature extractor
    feature_model_id = "demo_feature_extractor_001"
    feature_config = {
        'input_channels': 2,
        'action_dim': 15,
        'input_crop_size': 128
    }
    
    save_trained_model(
        model=feature_model,
        model_id=feature_model_id,
        architecture="vanilla_cnn",  # Update architecture
        model_config=feature_config,
        training_info={'task': 'feature_extraction'},
        task="feature_extraction",
        save_dir="saved_models"
    )
    
    # Create a hybrid model using the pretrained features
    print("🔀 Creating hybrid model with pretrained features...")
    hybrid_model = create_model_with_pretrained_features(
        architecture="vanilla_cnn",
        feature_extractor_model_id=feature_model_id,
        model_config={
            'input_channels': 2,
            'action_dim': 25,  # Different output size
            'input_crop_size': 128
        },
        save_dir="saved_models"
    )
    
    print("✅ Hybrid model created successfully!")
    
    # Test the hybrid model
    test_input = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        output = hybrid_model(test_input)
    print(f"   Hybrid model output shape: {output.shape}")


def demonstrate_model_registry():
    """Demonstrate the model registry functionality"""
    print("\n" + "=" * 60)
    print("📚 DEMONSTRATING MODEL REGISTRY")
    print("=" * 60)
    
    registry = ModelRegistry("saved_models")
    
    # List all models
    all_models = registry.list_models()
    print(f"📋 All registered models ({len(all_models)}):")
    for model_id in all_models:
        info = registry.get_model_info(model_id)
        print(f"   - {model_id}: {info.get('architecture')} ({info.get('task')})")
    
    # Filter by architecture
    cnn_models = registry.list_models(architecture="vanilla_cnn")
    print(f"\n🔍 CNN models ({len(cnn_models)}):")
    for model_id in cnn_models:
        print(f"   - {model_id}")
    
    # Filter by task
    bc_models = registry.list_models(task="behavior_cloning")
    print(f"\n🎯 Behavior cloning models ({len(bc_models)}):")
    for model_id in bc_models:
        print(f"   - {model_id}")


def demonstrate_torchscript_loading():
    """Demonstrate TorchScript model loading"""
    print("\n" + "=" * 60)
    print("⚡ DEMONSTRATING TORCHSCRIPT LOADING")
    print("=" * 60)
    
    # Check if we have any TorchScript models
    registry = ModelRegistry("saved_models")
    models = registry.list_models()
    
    for model_id in models:
        info = registry.get_model_info(model_id)
        if info.get('torchscript_available'):
            print(f"🔄 Loading TorchScript version of {model_id}...")
            
            # Load TorchScript model
            model_path = info['model_file']
            torchscript_model = ModelLoader.load_torchscript(model_path)
            
            print("✅ TorchScript model loaded successfully!")
            
            # Test the TorchScript model
            test_input = torch.randn(1, 2, 256, 256)
            with torch.no_grad():
                output = torchscript_model(test_input)
            print(f"   TorchScript output shape: {output.shape}")
            print("   This model can run without PyTorch source code!")
            break
    else:
        print("❌ No TorchScript models available")


def main():
    """Run all demonstrations"""
    print("🎬 MODEL UTILITIES DEMONSTRATION")
    print("This script demonstrates codebase-independent model persistence")
    print("and component reuse capabilities.\n")
    
    # Create save directory
    Path("saved_models").mkdir(exist_ok=True)
    
    # 1. Save a model
    model_id = demonstrate_model_saving()
    
    # 2. Load the model
    loaded_model, metadata = demonstrate_model_loading(model_id)
    
    # 3. Extract components
    hybrid_model = demonstrate_component_extraction(model_id)
    
    # 4. Create hybrid models
    demonstrate_hybrid_model_creation()
    
    # 5. Show registry functionality
    demonstrate_model_registry()
    
    # 6. Demonstrate TorchScript
    demonstrate_torchscript_loading()
    
    print("\n" + "=" * 60)
    print("✨ DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("Key benefits demonstrated:")
    print("• ✅ Codebase-independent model persistence")
    print("• ✅ Component extraction and reuse")
    print("• ✅ Hybrid model creation")
    print("• ✅ Model registry management")
    print("• ✅ TorchScript deployment readiness")
    print("• ✅ Full metadata preservation")


if __name__ == "__main__":
    main()
