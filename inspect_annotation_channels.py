"""
Inspect channels of the downloaded annotation file (Fixed).
"""
import tifffile
import numpy as np
import os
import glob

def inspect_file():
    # Find any ome.tiff file
    files = glob.glob("d:/AI/paper/CellSam/allen_data/*.ome.tiff")
    if not files:
        print("No files in allen_data")
        return
        
    file_path = files[0]
    print(f"Inspecting: {file_path}")
    
    with tifffile.TiffFile(file_path) as tif:
        print(f"Series count: {len(tif.series)}")
        series = tif.series[0]
        # Shape usually (Time, Channel, Z, Y, X)
        print(f"Shape: {series.shape}")
        print(f"Dtype: {series.dtype}")
        
        # Try to read channel names from OME metadata
        try:
            ome = tif.ome_metadata
            # Parsing simplisticly or just printing raw snippet
            if ome:
                print("\nOME Metadata snippet (checking for Name):")
                # Look for "Name" attributes in Channel tags
                import re
                names = re.findall(r'Name="([^"]+)"', ome)
                if names:
                    print("Channel Names found:", names)
                else:
                    print("No explicit channel names found in first scan.")
                    print(ome[:1000])
        except Exception as e:
            print(f"Could not read metadata: {e}")

if __name__ == "__main__":
    inspect_file()
