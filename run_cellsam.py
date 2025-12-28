import numpy as np
from cellSAM import segment_cellular_image
import matplotlib.pyplot as plt
import cv2
import os

# Set DeepCell Access Token (Provided by user)
os.environ["DEEPCELL_ACCESS_TOKEN"] = "X2Od0tJX.te0hEWOzZlRXoJzh5pkvw7l4S5GdpPxs"

def run_inference(image_path, device='cuda'):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # 1. Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image. Please check file format.")
        return

    # Convert BGR to RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Image shape: {img.shape}")
    print("Running Cell-SAM segmentation... This may take a moment.")

    # 2. Run segmentation
    try:
        # segment_cellular_image returns: mask, embedding, (optional extra)
        # We handle potentially 2 or 3 return values just in case
        result = segment_cellular_image(img, device=device)
        if len(result) == 3:
            mask, embedding, _ = result
        else:
            mask, embedding = result # Fallback if API changed
            
    except Exception as e:
        print(f"Error during segmentation: {e}")
        print("Tip: If using CUDA, ensure you have a compatible GPU and PyTorch installed.")
        print("Fallback to device='cpu' if necessary.")
        return

    # 3. Visualize results
    print("Segmentation complete. Displaying results...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmentation Mask (Cell-SAM)")
    # Using a high-contrast colormap
    plt.imshow(mask, cmap='nipy_spectral', interpolation='nearest') 
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can change the image path here or pass it via command line
    # For demonstration, we'll ask for input if not hardcoded
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Create a dummy image if no image works
        print("No image provided. To use your own image, run: python run_cellsam.py path/to/image.jpg")
        print("Input path to image:")
        img_path = input("Path: ").strip().strip('"') # Remove quotes if user drags file
    
    if img_path:
        # Check for CUDA availability
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
        except ImportError:
            device = 'cpu'
            print("PyTorch not found? Defaulting to device='cpu' string, but this will likely fail later.")

        run_inference(img_path, device=device)
