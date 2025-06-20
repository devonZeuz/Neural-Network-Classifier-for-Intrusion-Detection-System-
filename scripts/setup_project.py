#!/usr/bin/env python3
"""
Project Setup Script for IoT Intrusion Detection System

This script sets up the complete project structure and downloads necessary data
for the IoT Network Traffic Intrusion Detection System.

Author: Devon Cardoso Elias
Institution: Vistula University
Year: 2025
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import argparse


def create_directory_structure():
    """Create the complete project directory structure"""
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/interim",
        "models/saved_models",
        "models/checkpoints",
        "results/figures",
        "results/confusion_matrices",
        "results/performance_metrics",
        "logs/fit",
        "logs/tensorboard",
        "notebooks",
        "src",
        "tests",
        "docs",
        "scripts"
    ]
    
    print("Creating project directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    return True


def create_data_readme():
    """Create README for data directory"""
    
    readme_content = """# Data Directory

This directory contains the IoT-23 dataset and processed versions.

## Structure

- `raw/`: Original IoT-23 dataset files
- `processed/`: Preprocessed and cleaned data ready for training
- `interim/`: Intermediate processing steps

## IoT-23 Dataset

The IoT-23 dataset contains network traffic from IoT devices infected with malware.

### Download Instructions

1. Visit: https://www.stratosphereips.org/datasets-iot23
2. Download the light version (8.7 GB) or full version (20 GB)
3. Extract the files to the `raw/` directory

### Expected Structure

```
data/raw/
├── CTU-IoT-Malware-Capture-34-1/
│   └── bro/
│       └── conn.log.labeled
├── CTU-IoT-Malware-Capture-48-1/
│   └── bro/
│       └── conn.log.labeled
└── CTU-IoT-Malware-Capture-60-1/
    └── bro/
        └── conn.log.labeled
```

## Usage

Run the preprocessing script to prepare data for training:

```bash
python scripts/preprocess_data.py --input data/raw --output data/processed
```
"""
    
    with open("data/README.md", "w") as f:
        f.write(readme_content)
    
    print("  ✓ Created: data/README.md")


def create_gitignore():
    """Create comprehensive .gitignore file"""
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Dataset files (large files)
data/raw/*.csv
data/raw/*.log
data/raw/*.pcap
data/raw/*.zip
data/raw/*.tar.gz
data/processed/*.csv
data/processed/*.pkl
data/processed/*.h5

# Model files
models/saved_models/*.h5
models/saved_models/*.pkl
models/checkpoints/*

# Log files
logs/**/*.log
logs/fit/*
logs/tensorboard/*

# Results and outputs
results/figures/*.png
results/figures/*.jpg
results/figures/*.pdf
results/confusion_matrices/*
results/performance_metrics/*

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary files
*.tmp
*.temp
*.bak
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("  ✓ Created: .gitignore")


def create_license():
    """Create MIT License file"""
    
    license_content = """MIT License

Copyright (c) 2025 Devon Cardoso Elias

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open("LICENSE", "w") as f:
        f.write(license_content)
    
    print("  ✓ Created: LICENSE")


def install_dependencies():
    """Install required Python packages"""
    
    print("Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        print("  ✓ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error installing dependencies: {e}")
        return False


def create_example_notebook():
    """Create an example Jupyter notebook"""
    
    notebook_content = '''{...}'''
    
    with open("notebooks/example_analysis.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("  ✓ Created: notebooks/example_analysis.ipynb")


def main():
    """Main setup function"""
    
    parser = argparse.ArgumentParser(description="Setup IoT Intrusion Detection System project")
    parser.add_argument("--no-install", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--minimal", action="store_true",
                       help="Create minimal project structure only")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IoT Intrusion Detection System - Project Setup")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Create essential files
    create_gitignore()
    create_license()
    create_data_readme()
    
    if not args.minimal:
        create_example_notebook()
        
        # Install dependencies
        if not args.no_install:
            if os.path.exists("requirements.txt"):
                install_dependencies()
            else:
                print("  ! requirements.txt not found - skipping dependency installation")
    
    print("\n" + "=" * 60)
    print("Project setup completed successfully!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Download IoT-23 dataset to data/raw/ directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run example notebook: jupyter notebook notebooks/example_analysis.ipynb")
    print("4. Explore the src/ directory for implementation details")
    
    print("\nProject structure created:")
    print("├── data/                 # Dataset storage")
    print("├── models/               # Trained model storage")
    print("├── notebooks/            # Jupyter notebooks")
    print("├── src/                  # Source code")
    print("├── results/              # Results and figures")
    print("├── docs/                 # Documentation")
    print("├── scripts/              # Utility scripts")
    print("└── tests/                # Unit tests")
    
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main() 