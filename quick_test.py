import sys, os
print("🧪 Quick Test")
print(f"📍 Directory: {os.getcwd()}")
print(f"🐍 Python: {sys.version}")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    import sklearn, pandas, numpy
    print("✅ All packages work!")
    print("🎉 Setup successful!")
except Exception as e:
    print(f"❌ Error: {e}")
