# IoT Network Traffic Intrusion Detection System

**A Neural Network Classifier for Anomaly-Based Intrusion Detection in IoT Networks**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a deep learning-based Intrusion Detection System (IDS) specifically designed for IoT network traffic classification. Using the IoT-23 dataset, our neural network model achieves **83.66% accuracy** with a **ROC-AUC score of 0.9714** in distinguishing between benign and malicious network traffic.

## 🏆 Key Achievements

- **High Performance**: 83.66% accuracy with 0.9714 ROC-AUC score
- **Lightweight Architecture**: Only 18,246 trainable parameters  
- **Efficient Resource Usage**: 4.6% CPU utilization, 725.55 MB memory footprint
- **Real-time Capable**: Optimized for practical deployment scenarios

## 📊 Dataset

**IoT-23 Dataset** - Stratosphere Laboratory & Avast Software collaboration

- **Size**: 500+ hours of network traffic, 300+ million labeled flows
- **Scenarios**: 23 scenarios (20 malicious, 3 benign) from 2018-2019
- **Malware Types**: Mirai, Torii, Okiru botnet traffic
- **Used Captures**: CTU-IoT-Malware-Capture-34-1, 48-1, and 60-1

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/iot-intrusion-detection.git
cd iot-intrusion-detection

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Visit: https://www.stratosphereips.org/datasets-iot23
2. Download the IoT-23 dataset files
3. Extract to `data/raw/` directory

Expected structure:
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

### Running the Analysis

**Option 1: Jupyter Notebooks (Recommended)**
```bash
# Start Jupyter
jupyter notebook

# Run any of these notebooks:
# - notebooks/THE BEST FINAL Chatgpt 34-1.ipynb
# - notebooks/THE BEST FINAL Chatgpt 48-1.ipynb  
# - notebooks/THE BEST FINAL Chatgpt 60-1.ipynb
```

**Option 2: Python Scripts**
```bash
# Preprocess data
python src/data_preprocessing.py

# Train model
python src/training.py

# Evaluate model
python src/evaluation.py
```

## 🏗️ Architecture

### Neural Network Model
```
Input Layer (74 features)
    ↓
Flatten Layer
    ↓
Dense Layer (128 units, ReLU, L2 regularization)
    ↓
Dropout (40%)
    ↓
Dense Layer (64 units, ReLU, L2 regularization)
    ↓
Dropout (30%)
    ↓
Output Layer (6 classes, Softmax)
```

### Key Features
- **SMOTE** for handling class imbalance
- **RobustScaler** for feature scaling (outlier-resistant)
- **Stratified K-Fold** cross-validation (5 folds)
- **Early stopping** to prevent overfitting
- **Class weighting** for balanced training

## 📈 Results

| Metric | Value |
|--------|-------|
| Accuracy | 83.66% |
| ROC-AUC | 0.9714 |
| Precision (Benign) | 100% |
| Recall (Benign) | 99.94% |
| F1-Score (C&C-HeartBeat) | 98-99% |

### Class-Specific Performance
- **Benign Traffic**: Perfect detection (0.06% misclassification)
- **C&C-HeartBeat-FileDownload**: 100% precision and recall
- **DDoS Attacks**: High accuracy detection
- **PortScan Variants**: Lower performance (area for improvement)

## 📁 Project Structure

```
iot-intrusion-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/                    # Jupyter analysis notebooks
│   ├── THE BEST FINAL Chatgpt 34-1.ipynb
│   ├── THE BEST FINAL Chatgpt 48-1.ipynb
│   └── THE BEST FINAL Chatgpt 60-1.ipynb
├── src/                         # Source code
│   ├── data_preprocessing.py
│   ├── model_architecture.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
├── data/                        # Dataset storage
│   ├── raw/
│   └── processed/
├── models/                      # Trained models
│   └── saved_models/
├── docs/                        # Documentation
│   └── methodology.md
└── scripts/                     # Utility scripts
    ├── setup_project.py
    ├── download_data.py
    └── preprocess_data.py
```

## 🛠️ Usage Examples

### Basic Usage
```python
from src.data_preprocessing import IoTDataPreprocessor
from src.model_architecture import IoTIDSModel
from src.evaluation import IoTIDSEvaluator

# Preprocess data
preprocessor = IoTDataPreprocessor()
data = preprocessor.preprocess_pipeline("data/raw/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled")

# Train model
model = IoTIDSModel(input_shape=(data['num_features'],), num_classes=data['num_classes'])
trained_model = model.build_model()

# Evaluate
evaluator = IoTIDSEvaluator(model=trained_model, label_encoder=data['label_encoder'])
results = evaluator.generate_evaluation_report(X_test, y_test, save_dir="results/")
```

### Advanced Cross-Validation
```python
from src.training import IoTIDSTrainer

trainer = IoTIDSTrainer()
cv_results = trainer.cross_validate_model(
    data['X'], data['y'], 
    data['feature_names'], 
    data['label_encoder']
)
trainer.save_results("results/")
```

## 📚 Documentation

- **[Methodology](docs/methodology.md)**: Detailed research methodology and approach
- **[Jupyter Notebooks](notebooks/)**: Complete analysis workflows
- **[Source Code](src/)**: Modular implementation with full documentation

## 🎓 Academic Context

This project was developed as part of an engineering thesis at **Vistula University**, Faculty of Computer Engineering, Graphic Design and Architecture, under the supervision of **Dr. Mariusz Jakubowski**.

### Research Questions Addressed
1. How can the proposed model be optimized for real-time detection of anomalies in high-speed network environments?
2. What are the trade-offs between model accuracy and computational efficiency for practical deployment?

## 🔮 Future Work

- **Hierarchical Classification**: Specialized PortScan detection mechanisms
- **Ensemble Methods**: Combining multiple models for improved accuracy
- **Feature Engineering**: Advanced feature selection and extraction techniques
- **Real-time Optimization**: Further reduction in latency and resource usage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@thesis{elias2025iot,
  title={Implementation of a neural network classifier for an intrusion detection system},
  author={Devon Cardoso Elias},
  year={2025},
  school={Vistula University},
  type={Engineer's thesis},
  address={Warsaw, Poland}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repository if you found it helpful!**
