# IoT Network Traffic Intrusion Detection System
### A Neural Network Classifier for Anomaly-Based Intrusion Detection in IoT Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[![Accuracy](https://img.shields.io/badge/Accuracy-83.66%25-green.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9714-green.svg)]()
[![Parameters](https://img.shields.io/badge/Parameters-18.2K-blue.svg)]()
[![Memory](https://img.shields.io/badge/Memory-725MB-blue.svg)]()

---

🎯 Project Overview
This project implements a deep learning-based Intrusion Detection System (IDS) specifically designed for IoT network traffic classification. Using the IoT-23 dataset, our neural network model achieves 83.66% accuracy with a ROC-AUC score of 0.9714 in distinguishing between benign and malicious network traffic.

🏆 Key Achievements
High Performance: 83.66% accuracy with 0.9714 ROC-AUC score
Lightweight Architecture: Only 18,246 trainable parameters
Efficient Resource Usage: 4.6% CPU utilization, 725.55 MB memory footprint
Real-time Capable: Optimized for practical deployment scenarios
📊 Dataset
IoT-23 Dataset - Stratosphere Laboratory & Avast Software collaboration

Size: 500+ hours of network traffic, 300+ million labeled flows
Scenarios: 23 scenarios (20 malicious, 3 benign) from 2018-2019
Malware Types: Mirai, Torii, Okiru botnet traffic
Used Captures: CTU-IoT-Malware-Capture-34-1, 48-1, and 60-1
🏗️ Architecture
Neural Network Model
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
Key Features
SMOTE for handling class imbalance
RobustScaler for feature scaling (outlier-resistant)
Stratified K-Fold cross-validation (5 folds)
Early stopping to prevent overfitting
Class weighting for balanced training
🚀 Quick Start
Prerequisites
bash
pip install -r requirements.txt
Running the Models
bash
# Run analysis on CTU-IoT-Malware-Capture-34-1
jupyter notebook notebooks/CTU-IoT-34-1-Analysis.ipynb

# Run analysis on CTU-IoT-Malware-Capture-48-1
jupyter notebook notebooks/CTU-IoT-48-1-Analysis.ipynb

# Run analysis on CTU-IoT-Malware-Capture-60-1
jupyter notebook notebooks/CTU-IoT-60-1-Analysis.ipynb
📁 Project Structure
iot-intrusion-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── docs/
│   ├── thesis.pdf
│   ├── methodology.md
│   └── results-analysis.md
├── notebooks/
│   ├── CTU-IoT-34-1-Analysis.ipynb
│   ├── CTU-IoT-48-1-Analysis.ipynb
│   └── CTU-IoT-60-1-Analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_architecture.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
│   ├── saved_models/
│   └── checkpoints/
├── results/
│   ├── figures/
│   ├── confusion_matrices/
│   └── performance_metrics/
├── scripts/
│   ├── download_data.py
│   ├── preprocess_data.py
│   └── train_model.py
└── tests/
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_evaluation.py
🔬 Methodology
Data Preprocessing Pipeline
Data Loading: Tab-delimited IoT-23 conn.log files
Feature Engineering: One-hot encoding for categorical variables
Data Cleaning: Removal of irrelevant features (uid, timestamps, IP addresses)
Class Balancing: SMOTE oversampling for minority classes
Feature Scaling: RobustScaler for outlier-resistant normalization
Model Training Process
Stratified K-Fold Cross-Validation (5 folds)
Class Weight Computation for balanced learning
Early Stopping (patience=10, monitor='val_loss')
Adam Optimizer (learning_rate=0.001)
TensorBoard logging for training visualization
📈 Results
Performance Metrics
Metric	Value
Accuracy	83.66%
ROC-AUC	0.9714
Precision (Benign)	100%
Recall (Benign)	99.94%
F1-Score (C&C-HeartBeat)	98-99%
Class-Specific Performance
Benign Traffic: Perfect detection (0.06% misclassification)
C&C-HeartBeat-FileDownload: 100% precision and recall
DDoS Attacks: High accuracy detection
PortScan Variants: Lower performance (area for improvement)
Computational Efficiency
Parameters: 18,246 trainable parameters
Memory Usage: 725.55 MB
CPU Usage: 4.6%
Training Time: ~3-4ms/step
Inference Time: Real-time capable
🛠️ Technical Implementation
Key Technologies
Python 3.8+
TensorFlow/Keras for deep learning
Scikit-learn for preprocessing and metrics
Imbalanced-learn for SMOTE implementation
Pandas/NumPy for data manipulation
Matplotlib/Seaborn for visualization
Model Architecture Details
python
model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(n_unique_y_res, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
🎓 Academic Context
This project was developed as part of an engineering thesis at Vistula University, Faculty of Computer Engineering, Graphic Design and Architecture, under the supervision of Dr. Mariusz Jakubowski.

Research Questions Addressed
How can the proposed model be optimized for real-time detection of anomalies in high-speed network environments?
What are the trade-offs between model accuracy and computational efficiency for practical deployment?
Key Contributions
Lightweight Architecture: Efficient model suitable for resource-constrained environments
Balanced Performance: High accuracy with low false positive rates
Practical Deployment: Real-time capable with minimal computational overhead
Comprehensive Evaluation: Multi-dataset validation with detailed performance analysis
🔮 Future Work
Optimization Opportunities
Hierarchical Classification: Specialized PortScan detection mechanisms
Ensemble Methods: Combining multiple models for improved accuracy
Feature Engineering: Advanced feature selection and extraction techniques
Real-time Optimization: Further reduction in latency and resource usage
Potential Applications
IoT Security Gateways: Edge deployment for real-time protection
Network Monitoring: Integration with existing security infrastructure
Anomaly Detection: Extension to other network security domains
📄 Citation
If you use this work in your research, please cite:

bibtex
@thesis{elias2025iot,
  title={Implementation of a neural network classifier for an intrusion detection system},
  author={Devon Cardoso Elias},
  year={2025},
  school={Vistula University},
  type={Engineer's thesis},
  address={Warsaw, Poland}
}
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

📧 Contact
Devon Cardoso Elias

Student ID: 64238
Institution: Vistula University
Program: Computer Science
⭐ Star this repository if you found it helpful!


