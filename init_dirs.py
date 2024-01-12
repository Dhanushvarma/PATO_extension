import os

# Define the directory structure
dirs = [
    "HBC/models",
    "HBC/utils",
    "HBC/training",
    "HBC/inference",
    "HBC/tests"
]

# Define the files to be created in each directory
files = {
    "HBC/models": ["__init__.py", "encoder.py", "decoder.py", "vae.py"],
    "HBC/utils": ["__init__.py", "data_loader.py", "r3m_loader.py"],
    "HBC/training": ["__init__.py", "train_vae.py"],
    "HBC/inference": ["__init__.py", "generate_goals.py"],
    "HBC/tests": ["__init__.py", "test_encoder.py", "test_decoder.py", "test_vae.py"],
    "HBC": ["requirements.txt", "README.md"]
}

# Create directories
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

# Create files
for dir, files in files.items():
    for file in files:
        with open(os.path.join(dir, file), 'w') as fp:
            pass  # Just create the file

print("Project structure initialized successfully.")
