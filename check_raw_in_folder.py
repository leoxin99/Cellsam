"""
Check if raw images are in the same folder as annotations.
"""
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def check_folder():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    prefix = "aics/integrated_transcriptomics_structural_organization_hipsc_cm/2d_autocontrasted_fields_and_single_cells/rescaled_2D_fov_tiff_path/"
    
    print(f"Listing 100 files in {prefix}")
    
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
    
    annotation_files = []
    other_files = []
    
    if 'Contents' in resp:
        for obj in resp['Contents']:
            key = obj['Key']
            filename = key.split('/')[-1]
            
            if "annotation" in filename:
                annotation_files.append(filename)
            else:
                other_files.append(filename)
                
    print(f"\nFound {len(annotation_files)} annotation files.")
    print(f"Found {len(other_files)} other files.")
    
    if other_files:
        print("\nPossible raw files:")
        for f in other_files[:10]:
            print(f"  {f}")
            
    if annotation_files:
        print("\nExample annotation:")
        print(f"  {annotation_files[0]}")

if __name__ == "__main__":
    check_folder()
