#!/usr/bin/env python3
"""
Quick test script for Devon's IoT IDS setup
"""

import os
import sys

def test_environment():
    print("🧪 Testing IoT IDS Environment")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Check if we're in the right place
    expected_files = ['README.md', 'requirements.txt', 'src', 'notebooks']
    missing_files = []
    
    print("\n📁 Checking project files:")
    for file in expected_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - Missing!")
            missing_files.append(file)
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    
    # Test imports
    print("\n📦 Testing package imports:")
    packages = [
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]
    
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ❌ {name}: Not installed")
    
    # Test project imports
    print("\n🔍 Testing project modules:")
    sys.path.append('src')
    
    try:
        from data_preprocessing import IoTDataPreprocessor
        print("   ✅ data_preprocessing module")
    except ImportError as e:
        print(f"   ❌ data_preprocessing: {e}")
    
    try:
        from model_architecture import IoTIDSModel
        print("   ✅ model_architecture module")
    except ImportError as e:
        print(f"   ❌ model_architecture: {e}")
    
    try:
        from evaluation import IoTIDSEvaluator
        print("   ✅ evaluation module")
    except ImportError as e:
        print(f"   ❌ evaluation: {e}")
    
    # Check virtual environment
    print(f"\n🏠 Virtual environment: {'✅ Active' if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix else '❌ Not active'}")
    
    print("\n" + "=" * 50)
    
    if missing_files:
        print("⚠️  Some project files are missing. Make sure you're in the right directory!")
        print("📍 Expected directory: /home/devon-elias/Desktop/Neural Network Classifier for Intrusion Detection System")
    else:
        print("🎉 Environment test completed!")
        print("\n✨ Next steps:")
        print("1. Download IoT-23 dataset to data/raw/")
        print("2. Run: jupyter notebook")
        print("3. Open any of your notebooks in the notebooks/ folder")

if __name__ == "__main__":
    test_environment()
