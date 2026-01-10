"""
View and analyze Allen Cell annotation TIFF file.
Run this directly to explore the file structure.
"""
import tifffile
import numpy as np
from pathlib import Path

# 文件路径
ANNOTATION_FILE = "d:/AI/paper/CellSam/allen_data/00ca7adf_627998dd_5500000014_63X_20190816_S2_P30_C3_annotations_corrected_rescaled.ome.tiff"

def main():
    print("=" * 60)
    print("Allen Cell Annotation File Viewer")
    print("=" * 60)
    
    # 1. 读取文件
    print(f"\nLoading: {Path(ANNOTATION_FILE).name}")
    data = tifffile.imread(ANNOTATION_FILE)
    
    # 2. 基本信息
    print(f"\n📊 基本信息:")
    print(f"   Shape: {data.shape}")
    print(f"   dtype: {data.dtype}")
    print(f"   总通道数: {data.shape[0]}")
    print(f"   图像尺寸: {data.shape[1]} x {data.shape[2]}")
    
    # 3. 每个通道的分析
    print(f"\n📈 通道分析:")
    channel_names = [
        "Ch0: 可能是 Brightfield",
        "Ch1: 可能是 GFP (Alpha-actinin)",
        "Ch2: 可能是 DAPI (细胞核)",
        "Ch3: 可能是 其他荧光",
        "Ch4: 可能是 其他荧光",
        "Ch5: 空白通道",
        "Ch6: 细胞掩膜 (Cell Mask)",
        "Ch7: 细胞核掩膜 (Nuclear Mask)",
        "Ch8: 结构掩膜 (Structure)",
        "Ch9: 实例分割 (Instance Seg)"
    ]
    
    for i in range(data.shape[0]):
        channel = data[i]
        unique_vals = len(np.unique(channel))
        min_val, max_val = channel.min(), channel.max()
        non_zero = np.count_nonzero(channel)
        coverage = non_zero / channel.size * 100
        
        name = channel_names[i] if i < len(channel_names) else f"Ch{i}"
        print(f"   {name}")
        print(f"      唯一值: {unique_vals}, 范围: [{min_val}, {max_val}], 覆盖率: {coverage:.1f}%")
    
    # 4. 保存每个通道为单独的 PNG
    print(f"\n🖼️ 保存各通道为 PNG...")
    from skimage import io
    
    output_dir = Path("d:/AI/paper/CellSam/annotation_channels")
    output_dir.mkdir(exist_ok=True)
    
    for i in range(data.shape[0]):
        channel = data[i]
        # 归一化到 0-255
        if channel.max() > 0:
            normalized = ((channel - channel.min()) / (channel.max() - channel.min() + 1e-8) * 255).astype(np.uint8)
        else:
            normalized = channel.astype(np.uint8)
        
        output_path = output_dir / f"channel_{i:02d}.png"
        io.imsave(str(output_path), normalized)
        print(f"   Saved: {output_path.name}")
    
    print(f"\n✅ 分析完成! 图像保存到: {output_dir}")

if __name__ == "__main__":
    main()
