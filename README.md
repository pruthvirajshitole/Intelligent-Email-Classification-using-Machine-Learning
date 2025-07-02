# 📧 Spam Email Classification

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter"/>
</div>

## 🎯 Project Overview

This project implements a machine learning model to classify emails as spam or non-spam (ham). Using advanced natural language processing techniques and machine learning algorithms, the system provides accurate email classification to help users filter unwanted messages.

## 🔧 Technologies Used

- **Python**: Primary programming language
- **Natural Language Processing (NLP)**: For text preprocessing and feature extraction
- **scikit-learn**: Machine learning library for model development
- **Streamlit**: Web framework for UI deployment
- **Jupyter Notebook**: For model development and analysis
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 📊 Dataset

The project uses a comprehensive email dataset containing both spam and non-spam messages. The dataset is preprocessed and vectorized using natural language processing techniques to extract meaningful features for classification.

**Dataset Source:** [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Spam-Email-Classification.git
   cd Spam-Email-Classification
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 🔗 Live Demo
This application is deployed on Streamlit Cloud: [Spam Email Classifier on Streamlit.io](https://spam-email-classification-using-ml.streamlit.app/)


1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Access the application through your web browser at the URL provided in the terminal (usually `http://localhost:8501`)

3. Enter the email text you want to classify and submit

## 🔬 Model Development

The model development process is documented in `model.ipynb`. This notebook contains:
- Data preprocessing steps
- Feature engineering
- Model selection and training
- Performance evaluation
- Model serialization

## 📁 Project Structure

```
├── app.py              # Streamlit application
├── dataset.csv         # Training dataset
├── model.ipynb         # Model development notebook
├── model.pkl           # Trained model
├── requirements.txt    # Project dependencies
└── vectorizer.pkl      # Text vectorizer
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
<div align="center">
Made with ❤️ for a spam-free inbox
</div>
