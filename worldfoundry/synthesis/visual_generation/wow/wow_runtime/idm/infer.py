#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import json
import argparse
import h5py
from PIL import Image

# ----------------------
# Dynamically import model classes
# ----------------------
def get_model_class(model_name):
    if model_name == 'dinov2':
        from models.idm_models import DinoIDM
        return DinoIDM
    if model_name == 'dinov2_flow':
        from models.idm_models import Dino3DFlowIDM
        return Dino3DFlowIDM
    elif model_name == 'resnet':
        from models.idm_models import ResNetIDM
        return ResNetIDM
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def load_model_with_text_map(model_path, device='cuda', args=None):
    """Load checkpoint and extract text mapping."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract text mapping if present
    if 'text_map' in checkpoint:
        text_map = checkpoint['text_map']
    else:
        text_map = {"default": 0}
    
    # Create model instance
    num_text_tokens = len(text_map)
    ModelClass = get_model_class(args.model)
    model = ModelClass(output_dim=args.output_dim)

    # Load weights (handle both bare state_dict and checkpoint format)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    loadreport = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model load report: {loadreport}")
    model.eval()
    
    return model, text_map


def infer_from_h5_files_simple(h5_files, model_path, output_path=None, device='cuda', max_actions=None, args=None):
    """
    Run simple inference on a list of HDF5 files and return simplified joint predictions.
    """
    # Load model and text mapping
    model, text_map = load_model_with_text_map(model_path, device, args=args)
    model = model.to(device)
    
    # Image preprocessing - keep only ToTensor here.
    # Model-specific resizing/normalization is done inside model (timm transforms).
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    all_results = []
    
    total_files = len(h5_files)
    total_actions = 0
    
    for file_idx, h5_file in enumerate(h5_files):
        if not os.path.exists(h5_file):
            print(f"Skipping missing file: {h5_file}")
            continue
            
        try:
            with h5py.File(h5_file, 'r') as f:

                # Read text instruction (robust fallback if missing)
                try:
                    text = f['text'][()]
                    if isinstance(text, bytes):
                        text = text.decode('utf-8')
                    text = str(text).strip()
                except Exception:
                    text = None

                if text is not None and text in text_map:
                    instr_id = text_map[text]
                else:
                    # Fallback to first mapping entry or 0
                    if isinstance(text_map, dict) and len(text_map) > 0:
                        first_key = next(iter(text_map.keys()))
                        instr_id = text_map[first_key]
                    else:
                        instr_id = 0
                
                # Read images and actions
                images = f['observations']['image'][:]  # (T, H, W, 3)
                actions = f['action'][:]  # (T, 7)
                
                # Debug info
                print(f"File {file_idx+1}/{total_files}: {os.path.basename(h5_file)}")
                print(f"  Num images: {len(images)}")
                print(f"  Num actions: {len(actions)}")
                print(f"  Instruction text: {text}")
                
                if len(images) < 2:
                    print(f"  Skipping: not enough images")
                    continue
                
                # Limit number of processed actions per file if requested
                if max_actions is not None:
                    num_actions = min(max_actions, len(actions) - 1)
                else:
                    num_actions = len(actions) - 1
                
                print(f"  Will process {num_actions} actions")
                
                # Inference loop
                with torch.no_grad():
                    for i in range(1, num_actions + 1):
                        prev_image = images[i-1]
                        curr_image = images[i]
                        
                        prev_img = Image.fromarray(prev_image).convert('RGB')
                        curr_img = Image.fromarray(curr_image).convert('RGB')
                        
                        prev_tensor = transform(prev_img)
                        curr_tensor = transform(curr_img)
                        
                        img_pair = torch.stack([prev_tensor, curr_tensor], 0).unsqueeze(0).float().to(device)
                        instr = torch.tensor([instr_id]).to(device)
                        
                        # Prefer infer_forward if available, otherwise call forward
                        if hasattr(model, 'infer_forward'):
                            pred_action = model.infer_forward(img_pair, instr)
                        else:
                            pred_action = model.forward(img_pair, instr)
                        pred_action = pred_action.detach().squeeze().cpu().numpy().tolist()
                        
                        all_results.append({
                            "file": os.path.basename(h5_file),
                            "frame_index": i,
                            "arm_joints": pred_action
                        })
                
                total_actions += num_actions
                print(f"  Done, cumulative actions processed: {total_actions}")
                
        except Exception as e:
            print(f"  Error processing file {h5_file}: {str(e)}")
            continue
    
    print(f"\nTotal predicted actions: {len(all_results)}")
    
    # Save results if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Simple inference script')
    parser.add_argument('--h5_files', type=str, nargs='+', required=True, 
                       help='List of input h5 file paths')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, default=os.environ.get("INFER_OUTPUT", "simple_inference_results.json"), 
                       help='Output JSON file path (can be set via INFER_OUTPUT env var)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='CUDA computation device, for example cuda or cuda:0')
    parser.add_argument('--max_actions', type=int, default=None,
                       help='Maximum number of actions to process per H5 file')
    parser.add_argument('--output_dim', type=int, default=7, help='Action output dimension')
    parser.add_argument('--model', type=str, default='dinov2_flow', choices=['dinov2','dinov2_state','dinov2_flow', 'resnet'],
                    help='Choose model type (default: dinov2_flow)')
    
    args = parser.parse_args()
    
    if not args.device.startswith('cuda'):
        raise ValueError("WoW IDM inference requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WoW IDM inference.")
    
    try:
        results = infer_from_h5_files_simple(
            h5_files=args.h5_files,
            model_path=args.model_path,
            output_path=args.output_path,
            device=args.device,
            max_actions=args.max_actions,
            args=args
        )
        
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
