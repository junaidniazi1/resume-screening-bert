# 🎯 Resume Screening with DistilBERT

An **AI-powered Resume Screening System** that automatically classifies resumes into job categories using **DistilBERT**, a lightweight and efficient version of BERT. This project combines natural language processing with machine learning to streamline the recruitment process.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/transformers-4.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-orange.svg)

## 🚀 Features

- **Dual Model Architecture**: Compare BERT and DistilBERT performance
- **Text Preprocessing**: Comprehensive cleaning pipeline for resume text
- **Multi-class Classification**: Categorizes resumes into various job roles
- **Interactive Web Interface**: Real-time predictions via Streamlit dashboard
- **Model Persistence**: Save and reload trained models efficiently
- **Probability Analysis**: Get confidence scores for all categories
- **Performance Metrics**: Detailed classification reports and evaluation

## 📂 Project Structure

```
📦 Resume_Screening/
├── 📓 untitled2.py                    # Main training script (Jupyter notebook export)
├── 🎬 demo_video.mp4                  # Demo video showing the application
├── 🖥️ app.py                          # Streamlit web application
├── 📊 Resume.csv                      # Training dataset
├── 🤖 distilbert_saved/               # Saved DistilBERT model directory
│   ├── config.json                   # Model configuration
│   ├── model.safetensors             # Model weights
│   ├── tokenizer_config.json         # Tokenizer configuration
│   ├── vocab.txt                     # Vocabulary file
│   └── special_tokens_map.json       # Special tokens mapping
├── 🏷️ label_encoder.pkl               # Saved label encoder for categories
├── 📋 requirements.txt                # Python dependencies
└── 📖 README.md                       # Project documentation
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/resume-screening.git
cd resume-screening
```

### Step 2: Create Virtual Environment
```bash
python -m venv resume_env
source resume_env/bin/activate  # On Windows: resume_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Required Models
The script will automatically download BERT and DistilBERT models from Hugging Face on first run.

## 🏋️ Training the Model

### Data Preparation
The script expects a CSV file with the following columns:
- `Resume_str`: Raw resume text
- `Category`: Job category labels
- `ID`: Resume identifier
- `Resume_html`: HTML version (optional)

### Run Training Script
```python
python untitled2.py
```

### Training Process
1. **Data Loading**: Loads resume data from CSV
2. **Preprocessing**: 
   - Converts text to lowercase
   - Removes punctuation and numbers
   - Normalizes whitespace
3. **Model Training**:
   - BERT Base (6 epochs)
   - DistilBERT (4 epochs)
4. **Model Evaluation**: Classification reports and metrics
5. **Model Saving**: Saves the best performing model (DistilBERT)

### Key Training Parameters
```python
# BERT Training
num_train_epochs=6
per_device_train_batch_size=16
max_length=128

# DistilBERT Training  
num_train_epochs=4
per_device_train_batch_size=16
max_length=128
```

## 🎯 Model Architecture

### DistilBERT Configuration
- **Base Model**: `distilbert-base-uncased`
- **Sequence Length**: 128 tokens
- **Classification Head**: Multi-class classifier
- **Optimization**: AdamW optimizer with weight decay

### Preprocessing Pipeline
```python
def preprocess_text(text):
    text = text.lower()                                    # Lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)                       # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()              # Normalize whitespace
    return text
```

## 📊 Running the Web Application

### Start Streamlit Server
```bash
streamlit run app.py
```

### Application Features
- **Resume Input**: Paste resume text in the text area
- **Real-time Prediction**: Instant classification results
- **Probability Scores**: Confidence levels for all categories
- **Interactive UI**: User-friendly interface with clear results


## 📈 Model Performance

### Evaluation Metrics
The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall

### Sample Performance Output
```
Classification Report:
                    precision    recall  f1-score   support
Data Science         0.85      0.88      0.87       45
Software Engineer    0.92      0.89      0.91       52
Marketing           0.78      0.82      0.80       38
...
accuracy                                0.86      500
macro avg           0.85      0.86      0.85      500
weighted avg        0.86      0.86      0.86      500
```

## 🔧 Customization

### Adding New Categories
1. Update your training data with new categories
2. Retrain the model using the training script
3. The label encoder will automatically handle new categories




## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```



## 📋 Dependencies

### Core Libraries
```
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
streamlit>=1.12.0
```

### Development Libraries
```
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
```



## 📝 Usage Examples

### Programmatic Usage
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_saved')
model = DistilBertForSequenceClassification.from_pretrained('./distilbert_saved')
label_encoder = joblib.load('label_encoder.pkl')

# Make prediction
resume_text = "Experienced software engineer with Python expertise..."
inputs = tokenizer(resume_text, return_tensors='pt', truncation=True, padding=True)
outputs = model(**inputs)
predicted_class = label_encoder.inverse_transform([outputs.logits.argmax().item()])[0]
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
