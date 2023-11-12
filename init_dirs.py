import os

# Define the directory structure
dirs = [
    "subgoal_predictor/models",
    "subgoal_predictor/utils",
    "subgoal_predictor/training",
    "subgoal_predictor/inference",
    "subgoal_predictor/tests"
]

# Define the files to be created in each directory
files = {
    "subgoal_predictor/models": ["__init__.py", "encoder.py", "decoder.py", "vae.py"],
    "subgoal_predictor/utils": ["__init__.py", "data_loader.py", "r3m_loader.py"],
    "subgoal_predictor/training": ["__init__.py", "train_vae.py"],
    "subgoal_predictor/inference": ["__init__.py", "generate_goals.py"],
    "subgoal_predictor/tests": ["__init__.py", "test_encoder.py", "test_decoder.py", "test_vae.py"],
    "subgoal_predictor": ["requirements.txt", "README.md"]
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
