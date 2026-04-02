import os
import urllib.request
import zipfile
import shutil

# Config - Google's stable dataset URL
URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
ZIP_NAME = "temp_data.zip"
EXTRACT_FOLDER = "temp_extracted"
FINAL_DIR = "PetImages"

def download_and_setup():
    # 1. Clean up old mess if exists
    if os.path.exists(FINAL_DIR):
        print(f"Removing old {FINAL_DIR}...")
        shutil.rmtree(FINAL_DIR)
    
    # 2. Download
    print(f"Downloading dataset from Google (68MB)...")
    urllib.request.urlretrieve(URL, ZIP_NAME)
    print("Download complete.")

    # 3. Unzip
    print("Unzipping...")
    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)

    # 4. Create your missing folders
    os.makedirs(os.path.join(FINAL_DIR, "Cat"), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DIR, "Dog"), exist_ok=True)

    # 5. Move files to the right place
    print("Organizing files...")
    base_source = os.path.join(EXTRACT_FOLDER, "cats_and_dogs_filtered")
    
    for sub in ["train", "validation"]:
        for animal in ["cats", "dogs"]:
            source_path = os.path.join(base_source, sub, animal)
            dest_animal = "Cat" if animal == "cats" else "Dog"
            dest_path = os.path.join(FINAL_DIR, dest_animal)
            
            for file_name in os.listdir(source_path):
                full_file_name = os.path.join(source_path, file_name)
                shutil.move(full_file_name, dest_path)

    # 6. Cleanup
    os.remove(ZIP_NAME)
    shutil.rmtree(EXTRACT_FOLDER)
    print("Done! 'PetImages' folder has been created.")

if __name__ == "__main__":
    download_and_setup()