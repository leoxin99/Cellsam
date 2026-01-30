"""
Analyze channel statistics to derive data-driven preprocessing parameters.
Outputs recommendations for:
1. Actn2 percentile range
2. CLAHE clip limit
3. DAPI Gaussian sigma
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_channels(data_dir: str, max_samples: int = 50):
    """Analyze pixel distributions for each channel."""
    
    data_path = Path(data_dir)
    image_dir = data_path / "images"
    
    npy_files = sorted(image_dir.glob("*.npy"))[:max_samples]
    print(f"Analyzing {len(npy_files)} samples from {image_dir}...")
    
    # Collect statistics
    ch0_all = []  # BF
    ch1_all = []  # DAPI
    ch2_all = []  # Actn2
    
    for f in tqdm(npy_files, desc="Loading"):
        img = np.load(f)
        
        # Handle (3, H, W) or (H, W, 3)
        if img.shape[0] == 3:
            ch0 = img[0].flatten()
            ch1 = img[1].flatten()
            ch2 = img[2].flatten()
        else:
            ch0 = img[..., 0].flatten()
            ch1 = img[..., 1].flatten()
            ch2 = img[..., 2].flatten()
        
        ch0_all.append(ch0)
        ch1_all.append(ch1)
        ch2_all.append(ch2)
    
    ch0_all = np.concatenate(ch0_all)
    ch1_all = np.concatenate(ch1_all)
    ch2_all = np.concatenate(ch2_all)
    
    print(f"\nTotal pixels analyzed: {len(ch0_all):,}")
    
    # === Ch0: BF (Brightfield) ===
    print("\n" + "="*50)
    print("Ch0: BF (Brightfield)")
    print("="*50)
    print(f"  Min: {ch0_all.min()}")
    print(f"  Max: {ch0_all.max()}")
    print(f"  Mean: {ch0_all.mean():.1f}")
    print(f"  Std: {ch0_all.std():.1f}")
    for p in [0.1, 0.5, 1, 2, 50, 98, 99, 99.5, 99.9]:
        print(f"  P{p}: {np.percentile(ch0_all, p):.1f}")
    
    # === Ch1: DAPI ===
    print("\n" + "="*50)
    print("Ch1: DAPI (Nuclei)")
    print("="*50)
    print(f"  Min: {ch1_all.min()}")
    print(f"  Max: {ch1_all.max()}")
    print(f"  Mean: {ch1_all.mean():.1f}")
    print(f"  Std: {ch1_all.std():.1f}")
    for p in [0.1, 0.5, 1, 2, 50, 98, 99, 99.5, 99.9]:
        print(f"  P{p}: {np.percentile(ch1_all, p):.1f}")
    
    # Estimate best sigma based on typical nucleus size
    # Nucleus diameter ~113px → radius ~56px → sigma for smoothing = radius / 3 ≈ 18?
    # But for mild smoothing, sigma = 1-3 is typical
    nonzero = ch1_all[ch1_all > np.percentile(ch1_all, 90)]  # Focus on nuclei
    print(f"  [Info] Non-zero ratio (>P90): {len(nonzero)/len(ch1_all)*100:.1f}%")
    
    # === Ch2: Actn2 (α-actinin) ===
    print("\n" + "="*50)
    print("Ch2: Actn2 (α-actinin)")
    print("="*50)
    print(f"  Min: {ch2_all.min()}")
    print(f"  Max: {ch2_all.max()}")
    print(f"  Mean: {ch2_all.mean():.1f}")
    print(f"  Std: {ch2_all.std():.1f}")
    for p in [0.1, 0.5, 1, 2, 50, 98, 99, 99.5, 99.9]:
        print(f"  P{p}: {np.percentile(ch2_all, p):.1f}")
    
    # Dynamic range analysis
    dr_p1_99 = np.percentile(ch2_all, 99) - np.percentile(ch2_all, 1)
    dr_p05_995 = np.percentile(ch2_all, 99.5) - np.percentile(ch2_all, 0.5)
    print(f"  Dynamic range P1-P99: {dr_p1_99:.1f}")
    print(f"  Dynamic range P0.5-P99.5: {dr_p05_995:.1f}")
    
    # === Recommendations ===
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    
    # Actn2: Use P1-P99 if dynamic range is similar to P0.5-P99.5
    if dr_p05_995 / dr_p1_99 < 1.1:
        print(f"  Actn2 percentile: P1-P99 (recommended, similar DR)")
    else:
        print(f"  Actn2 percentile: P0.5-P99.5 (current, keeps more detail)")
    
    # BF: CLAHE clip should be adjusted based on contrast
    bf_contrast = np.percentile(ch0_all, 99) - np.percentile(ch0_all, 1)
    print(f"  BF contrast (P1-P99): {bf_contrast:.1f}")
    if bf_contrast < 100:
        print(f"  CLAHE clip: 3.0 (higher, low contrast image)")
    else:
        print(f"  CLAHE clip: 2.0 (standard)")
    
    # DAPI: sigma based on typical nucleus size
    # Nucleus diameter ~113px → for mild smoothing, sigma = 1-2 is typical
    print(f"  DAPI sigma: 1.0-2.0 (mild smoothing recommended)")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    analyze_channels("d:/AI/paper/CellSam/data/processed")
