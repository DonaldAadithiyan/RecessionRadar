import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Dataset info
KAGGLE_DATASET = "shubhaanshkumar/us-recession-dataset"
OUTPUT_DIR = "data/raw"
CSV_NAME = "us_recession_data.csv"  # Update this if the actual file has a different name

def download_kaggle_dataset(dataset: str, output_dir: str):
    print("⏳ Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    os.makedirs(output_dir, exist_ok=True)

    print(f"⬇️ Downloading dataset: {dataset}")
    api.dataset_download_files(dataset, path=output_dir, unzip=True)

    print("✅ Dataset downloaded and extracted.")

# def verify_csv_exists(output_dir: str, file_name: str):
#     csv_path = os.path.join(output_dir, file_name)
#     if os.path.exists(csv_path):
#         print(f"📁 CSV file ready at: {csv_path}")
#     else:
#         print("⚠️ CSV file not found. Please check the file name in the script.")

if __name__ == "__main__":
    download_kaggle_dataset(KAGGLE_DATASET, OUTPUT_DIR)
    # verify_csv_exists(OUTPUT_DIR, CSV_NAME)
