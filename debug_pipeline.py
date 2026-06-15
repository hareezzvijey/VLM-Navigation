# debug_pipeline.py
import sys
import os

print("="*50)
print("DEBUG: Starting pipeline debug")
print("="*50)

# 1. Check current directory
print(f"\n1. Current directory: {os.getcwd()}")

# 2. Check if paths exist
print(f"\n2. Checking paths:")
parent_dir = os.path.dirname(os.getcwd())
grounding_path = os.path.join(parent_dir, 'GroundingDINO')
efficient_path = os.path.join(parent_dir, 'efficientvit')

print(f"   GroundingDINO path: {grounding_path}")
print(f"   Exists: {os.path.exists(grounding_path)}")
print(f"   EfficientViT path: {efficient_path}")
print(f"   Exists: {os.path.exists(efficient_path)}")

# 3. Try to import without adding paths first
print(f"\n3. Testing imports:")
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"   ❌ PyTorch import failed: {e}")

try:
    import cv2
    print(f"   ✅ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"   ❌ OpenCV import failed: {e}")

# 4. Try adding paths and importing
print(f"\n4. Adding custom paths and testing:")
sys.path.append(grounding_path)
sys.path.append(efficient_path)

try:
    from groundingdino.util.inference import load_model
    print(f"   ✅ GroundingDINO import successful")
except ImportError as e:
    print(f"   ❌ GroundingDINO import failed: {e}")

try:
    from efficientvit.sam_model_zoo import create_sam_model
    print(f"   ✅ EfficientViT import successful")
except ImportError as e:
    print(f"   ❌ EfficientViT import failed: {e}")

print("\n" + "="*50)
print("Debug complete")