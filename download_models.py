import os
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "AshokXavier/tumor_vision_models"
MODEL_DIR = "models"
MODELS = [
    "ensemble_2d.keras",
    "3d_model.keras"
]

def download_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")

    print(f"Starting model download from {REPO_ID}...")
    
    for filename in MODELS:
        print(f"Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            raise e

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
