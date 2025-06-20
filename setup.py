#!/usr/bin/env python3
"""
Setup script for IoT Network Traffic Intrusion Detection System

Author: Devon Cardoso Elias
Institution: Vistula University
Year: 2025
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="iot-intrusion-detection",
    version="1.0.0",
    author="Devon Cardoso Elias",
    author_email="your.email@example.com",  # Replace with your email
    description="A Neural Network Classifier for Anomaly-Based Intrusion Detection in IoT Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/iot-intrusion-detection",  # Replace with your repo URL
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/iot-intrusion-detection/issues",
        "Documentation": "https://github.com/yourusername/iot-intrusion-detection/blob/main/docs/methodology.md",
        "Source Code": "https://github.com/yourusername/iot-intrusion-detection",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iot-ids-preprocess=scripts.preprocess_data:main",
            "iot-ids-train=scripts.train_model:main",
            "iot-ids-setup=scripts.setup_project:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "intrusion detection",
        "iot security", 
        "neural networks",
        "machine learning",
        "cybersecurity",
        "network traffic analysis",
        "anomaly detection",
    ],
    zip_safe=False,
) 