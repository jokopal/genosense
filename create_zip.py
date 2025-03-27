"""
Script untuk membuat file zip dari aplikasi Genosense
"""

import os
import zipfile
from datetime import datetime

def create_zip():
    """Buat file zip berisi seluruh aplikasi"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"genosense_{timestamp}.zip"
    
    # File dan direktori yang akan dimasukkan ke dalam zip
    include_paths = [
        # File Python inti
        "app.py", "main.py", "models.py", "routes.py", 
        "cellular_automata.py", "ml_models.py", "setup_db.py",
        
        # File konfigurasi
        "dependencies.txt", "README.md", ".gitignore",
        ".github/workflows/python-app.yml",
        
        # Direktori statis
        "static", 
        
        # Template
        "templates",
        
        # Folder uploads (kosong)
        "uploads/.gitkeep"
    ]
    
    # File dan direktori yang akan diabaikan
    exclude_patterns = [
        "__pycache__", 
        "*.pyc", 
        "instance", 
        "venv", 
        ".git", 
        "*.db", 
        "*.sqlite",
        "uv.lock",
        "replit.nix",
        ".replit",
        "pyproject.toml"
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if os.path.isfile(path):
                # Tambahkan file
                zipf.write(path)
                print(f"Added file: {path}")
            elif os.path.isdir(path):
                # Tambahkan direktori dan semua file di dalamnya
                for root, dirs, files in os.walk(path):
                    # Periksa apakah direktori harus diabaikan
                    skip_dir = False
                    for pattern in exclude_patterns:
                        if pattern in root:
                            skip_dir = True
                            break
                    
                    if skip_dir:
                        continue
                    
                    # Tambahkan file-file dalam direktori
                    for file in files:
                        # Periksa apakah file harus diabaikan
                        skip_file = False
                        for pattern in exclude_patterns:
                            if "*" in pattern and pattern.replace("*", "") in file:
                                skip_file = True
                                break
                            elif pattern == file:
                                skip_file = True
                                break
                        
                        if skip_file:
                            continue
                        
                        file_path = os.path.join(root, file)
                        zipf.write(file_path)
                        print(f"Added file: {file_path}")
    
    print(f"\nZip file created: {zip_filename}")
    print(f"Size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
    
    return zip_filename

if __name__ == "__main__":
    zip_file = create_zip()
    print(f"\nDownload the zip file: {zip_file}")
    print("You can run this application in your GitHub repository using GitHub Actions.")
    print("Follow the instructions in README.md for setup and configuration.")