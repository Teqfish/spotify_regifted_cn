import os
import zipfile
import pandas as pd

# Upload directory
upload_dir = os.path.expanduser("~/GITHUB_UPLOAD_LOCATION_HERE")

# Search for the zipped file on Github
zipped_dir = None
for file in os.listdir(upload_dir):
    if file.lower().endswith('.zip') and 'spotify' in file.lower():
        zipped_dir = os.path.join(upload_dir, file)
        print(f"Found zip file: {zipped_dir}")
        break

if zipped_dir is None:
    raise FileNotFoundError("No matching zip files found.")

# Define unzipped directory path
unzipped_dir = os.path.splitext(zipped_dir)[0]  # removes .zip

# Unzipping the file
zf = zipfile.ZipFile(zipped_dir)
zf.extractall(unzipped_dir)
print(f"Extracted to: {unzipped_dir}")

# Empty list of json dfs
dfs = []

# Search unzipped folder for jsons containing "audio"
for root, dirs, files in os.walk(unzipped_dir):
    for file in files:
        if file.lower().endswith('.json') and 'audio' in file.lower():
            file_path = os.path.join(root, file)
            print(f"Reading: {file_path}")
            
            # Convert to DataFrame
            try:
                df = pd.read_json(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")