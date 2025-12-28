"""
List files to understand naming structure.
"""
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def list_sample():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    prefix = "aics/integrated_transcriptomics_structural_organization_hipsc_cm/"
    
    print(f"Listing 20 files in {prefix}")
    
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50)
    
    if 'Contents' in resp:
        for obj in resp['Contents']:
            # Show only root level files or interesting ones
            print(obj['Key'])

if __name__ == "__main__":
    list_sample()
