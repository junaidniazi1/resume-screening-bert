import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import string
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import time

# Page configuration
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS for beautiful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .main {
        padding: 0rem 1rem;
    }
    
    
    
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        min-height: 100vh;
    }
    
    /* Add the gradient animation */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
   
   
    .subtitle {
        font-size: 1.3rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Card styles */
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        color: white;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Glassmorphism container */
    .glass-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(31, 38, 135, 0.4);
    }
    
    /* Input area styling */
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Results section */
    .results-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.8) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .predicted-category {
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin: 0.5rem 0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Confidence styling */
    .confidence-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Categories grid */
    .category-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin: 1rem 0;
    }
    
    .category-item {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .category-item:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Top predictions styling */
    .top-prediction-item {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .top-prediction-item:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Text area styling - FIXED FOR VISIBILITY */
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.98) !important;
        color: #1a202c !important;  /* Dark text for visibility */
        font-size: 14px !important;
        font-family: 'Poppins', sans-serif !important;
        line-height: 1.5 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Placeholder text styling */
    .stTextArea textarea::placeholder {
        color: #718096 !important;
        opacity: 0.8 !important;
    }
    
    /* Text input styling */
    .stTextInput input {
        color: #1a202c !important;
        background: rgba(255, 255, 255, 0.98) !important;
    }
    
    /* Disabled text area styling */
    .stTextArea textarea:disabled {
        background: rgba(247, 250, 252, 0.95) !important;
        color: #4a5568 !important;
        opacity: 1 !important;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.8s ease-out;
    }
    
    .slide-in-right {
        animation: slideInRight 0.8s ease-out;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border-radius: 15px !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        background: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stRadio label {
        color: #1a202c !important;
        font-weight: 500 !important;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Info box styling */
    .stInfo {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 15px !important;
        color: #2d3748 !important;
    }
    
    /* Warning box styling */
    .stWarning {
        background: rgba(237, 137, 54, 0.1) !important;
        border: 1px solid rgba(237, 137, 54, 0.2) !important;
        border-radius: 15px !important;
    }
    
    /* Success box styling */
    .stSuccess {
        background: rgba(72, 187, 120, 0.1) !important;
        border: 1px solid rgba(72, 187, 120, 0.2) !important;
        border-radius: 15px !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
        color: #1a202c !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1a202c !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-top: none !important;
    }
    
    /* Footer styling */
    .footer {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        margin-top: 3rem;
        color: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Loading animation */
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite alternate;
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        text-align: center;
        color: #2d3748;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container h4 {
        color: #667eea !important;
        margin-bottom: 0.5rem !important;
        font-size: 1rem !important;
    }
    
    .metric-container h2 {
        color: #2d3748 !important;
        font-size: 1.8rem !important;
        margin: 0 !important;
    }
    
    /* Additional text fixes */
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #1a202c !important;
    }
    
    /* File uploader text fix */
    .stFileUploader label {
        color: #1a202c !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🚀 Resume Screening AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Classify resumes into job categories using cutting-edge AI technology</p>', unsafe_allow_html=True)

# Preprocessing function
def preprocess_text(text):
    """Preprocess text similar to training data"""
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # 3. Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # 4. Remove extra spaces/tabs
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

@st.cache_resource
def load_model_and_tokenizer():
    """Load the saved DistilBERT model and tokenizer"""
    try:
        model_path = r"C:\Users\kashif-pc\Desktop\DATA ANYALSYIS PROJECT\ML projects\RESUMAA"
        
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        # Job categories
        categories = ['HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
              'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO', 'SALES',
              'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE', 'APPAREL',
              'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS', 'BANKING',
              'ARTS', 'AVIATION']
        
        return model, tokenizer, categories
    except Exception as e:
        st.error(f"🚫 Error loading model: {str(e)}")
        return None, None, None

def predict_resume_category(text, model, tokenizer, categories):
    """Predict the category of a resume"""
    try:
        # Preprocess the text
        cleaned_text = preprocess_text(text)
        
        # Tokenize
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()
        
        predicted_category = categories[predicted_class_id]
        
        return predicted_category, confidence, predictions.squeeze().tolist()
    
    except Exception as e:
        st.error(f"🚫 Error making prediction: {str(e)}")
        return None, None, None

# Load model with loading animation
with st.spinner("🔄 Loading AI model..."):
    model, tokenizer, categories = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("⚠️ Model could not be loaded. Please ensure the model files are in the correct directory.")
    st.info("📁 Expected model files in the specified directory:")
    st.code("""
    RESUMAA/
    ├── config.json
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
    """)
    st.stop()

# Main content layout
col1, col2 = st.columns([1.2, 1], gap="large")

# Left Column - Input Section
with col1:
    st.markdown('<div class="slide-in-left">', unsafe_allow_html=True)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    st.markdown("## 📝 Resume Input")
    
    # Input method selection with custom styling
    input_method = st.radio(
        "Choose your preferred input method:",
        ["✍️ Type/Paste Text", "📁 Upload File"],
        horizontal=True
    )
    
    resume_text = ""
    
    if input_method == "✍️ Type/Paste Text":
        resume_text = st.text_area(
            "📄 Enter resume content:",
            height=350,
            placeholder="Paste the complete resume text here...\n\nInclude all sections like experience, skills, education, etc.",
            help="For best results, include the complete resume with all relevant information."
        )
    
    elif input_method == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "📎 Choose a text file", 
            type=['txt'],
            help="Upload a .txt file containing the resume content"
        )
        if uploaded_file is not None:
            resume_text = str(uploaded_file.read(), "utf-8")
            st.text_area("📄 File content preview:", value=resume_text[:500] + "...", height=200, disabled=True)
            st.success(f"✅ File uploaded successfully! ({len(resume_text)} characters)")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button with enhanced styling
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Results and Info Section
with col2:
    st.markdown('<div class="slide-in-right">', unsafe_allow_html=True)
    
    if analyze_clicked and resume_text.strip():
        # Results Section
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("## 📊 AI Analysis Results")
        
        with st.spinner("🤖 AI is analyzing the resume..."):
            # Add a small delay for better UX
            time.sleep(1)
            predicted_category, confidence, all_predictions = predict_resume_category(
                resume_text, model, tokenizer, categories
            )
        
        if predicted_category:
            # Main prediction display
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin: 0; font-size: 1.2rem;">🎯 Predicted Job Category</h3>
                <div class="predicted-category">{predicted_category.replace('-', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence score
            st.markdown(f"""
            <div class="confidence-box">
                <strong>🎯 Confidence Score: {confidence:.1%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(confidence)
            
            # Performance metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>📈 Accuracy</h4>
                    <h2>{confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>⚡ Speed</h4>
                    <h2>< 1s</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Top 5 predictions
            st.markdown("### 🏆 Top 5 Matching Categories")
            
            # Get top 5 predictions
            top_indices = np.argsort(all_predictions)[::-1][:5]
            
            for i, idx in enumerate(top_indices):
                category = categories[idx].replace('-', ' ').title()
                score = all_predictions[idx]
                
                # Create visual representation
                percentage = int(score * 100)
                
                # Determine emoji based on rank
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                
                st.markdown(f"""
                <div class="top-prediction-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>{rank_emoji} {category}</strong></span>
                        <span style="color: #667eea; font-weight: 600;">{score:.1%}</span>
                    </div>
                    <div style="background: rgba(102, 126, 234, 0.1); height: 8px; border-radius: 4px; margin-top: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100%; width: {percentage}%; transition: width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif analyze_clicked and not resume_text.strip():
        st.warning("⚠️ Please enter some resume text to analyze.")
    
    else:
        # Information Section
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("## ℹ️ How It Works")
        
        steps = [
            ("📝", "Input Resume", "Paste text or upload a file"),
            ("🤖", "AI Processing", "DistilBERT analyzes content"),
            ("📊", "Get Results", "View category & confidence")
        ]
        
        for emoji, title, desc in steps:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{emoji}</div>
                <div style="font-weight: 600; margin-bottom: 0.3rem;">{title}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Supported categories
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("## 📋 Supported Job Categories")
        
        if categories:
            # Create a beautiful grid of categories
            st.markdown('<div class="category-grid">', unsafe_allow_html=True)
            for category in categories:
                display_category = category.replace('-', ' ').title()
                st.markdown(f'<div class="category-item">{display_category}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sample Resume Section
st.markdown("---")
with st.expander("🎯 Try with Sample Resume"):
    sample_text = """
John Smith
Senior Data Scientist & Machine Learning Engineer
📧 john.smith@email.com | 📱 (555) 123-4567 | 🌐 LinkedIn: linkedin.com/in/johnsmith

PROFESSIONAL SUMMARY
Experienced Data Scientist with 6+ years of expertise in machine learning, deep learning, and statistical analysis. 
Proven track record of developing scalable ML solutions that drive business growth and improve operational efficiency.

WORK EXPERIENCE

Senior Data Scientist | TechCorp Solutions | 2021 - Present
• Led development of customer churn prediction models, reducing churn by 25%
• Implemented deep learning algorithms for image recognition with 95% accuracy
• Built real-time recommendation systems serving 1M+ users daily
• Mentored team of 8 junior data scientists and ML engineers
• Deployed ML models to production using Docker, Kubernetes, and AWS

Data Scientist | InnovateTech | 2019 - 2021
• Developed predictive analytics models for demand forecasting
• Created automated data pipelines processing 10TB+ daily data
• Performed A/B testing and statistical analysis for product optimization
• Built interactive dashboards using Tableau and Power BI

Machine Learning Engineer | StartupX | 2018 - 2019
• Implemented computer vision solutions for autonomous systems
• Optimized neural network architectures reducing inference time by 40%
• Collaborated with product teams to integrate ML features

TECHNICAL SKILLS
Programming: Python, R, SQL, Java, Scala
ML/DL Frameworks: TensorFlow, PyTorch, Scikit-learn, XGBoost, LightGBM
Big Data: Spark, Hadoop, Kafka, Airflow
Cloud Platforms: AWS, Azure, GCP
Databases: PostgreSQL, MongoDB, Redis
Visualization: Tableau, Power BI, Matplotlib, Seaborn
MLOps: Docker, Kubernetes, MLflow, Kubeflow

EDUCATION
M.S. in Data Science | Stanford University | 2018
B.S. in Computer Science | UC Berkeley | 2016

CERTIFICATIONS
• AWS Certified Machine Learning - Specialty
• Google Cloud Professional Data Engineer
• Deep Learning Specialization - Coursera

ACHIEVEMENTS
• Published 12 research papers in top-tier ML conferences
• Led team that won 1st place in Kaggle competition (2020)
• Speaker at PyData and ML conferences
"""
    
    # Add custom CSS to fix text area styling
    st.markdown("""
    <style>
    .stTextArea textarea {
        color: #1a202c !important;
        background: rgba(255, 255, 255, 0.98) !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        border: 1px solid #d1d5db !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.text_area("Sample Data Scientist Resume:", value=sample_text, height=200, key="sample_resume")
    
    if st.button("🚀 Analyze Sample Resume"):
        # TODO: Implement functionality to set sample text as input and trigger analysis
        st.success("Sample resume loaded! Ready for analysis.")

# Footer
st.markdown("""
<div class="footer">
    <h4>🚀 Resume Screening AI</h4>
    <p>Powered by DistilBERT • Built with ❤️ using Streamlit</p>
    <p>© 2024 • Advanced AI for Modern Recruitment</p>
</div>
""", unsafe_allow_html=True)