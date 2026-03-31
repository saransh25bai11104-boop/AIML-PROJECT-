# 📧 Spam Email Classifier — AI/ML Capstone Project

A machine learning project that automatically classifies emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and the Naive Bayes algorithm.

---

## 🎯 What This Project Does

This project builds and trains a text classification model that:
- Accepts raw email text as input
- Processes it using **TF-IDF vectorization**
- Predicts whether the email is **SPAM** or **HAM** using a **Multinomial Naive Bayes** classifier
- Outputs the prediction along with confidence probabilities

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core programming language |
| scikit-learn | Machine learning (TF-IDF, Naive Bayes, metrics) |
| pandas | Data manipulation |
| numpy | Numerical computation |
| pickle | Model persistence (save/load) |

---

## 📁 Project Structure

```
spam-classifier/
│
├── spam_classifier.py   # Core ML logic: data, training, evaluation, prediction
├── app.py               # Interactive command-line app for live predictions
├── tests.py             # Unit tests for all components
├── requirements.txt     # Python dependencies
├── model/               # Saved model files (auto-created after training)
│   ├── model.pkl
│   └── vectorizer.pkl
└── README.md            # This file
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1 — Train and evaluate the model

This trains the classifier, evaluates it, and saves it to disk.

```bash
python spam_classifier.py
```

**Expected output:**
```
Dataset loaded: 40 emails (20 spam, 20 ham)
Training Naive Bayes classifier with TF-IDF features...
Training complete!

Model Evaluation Results
Accuracy : 70.00%
...
Model saved to ./model/
```

### Option 2 — Interactive prediction mode

Classify your own email text in real time.

```bash
python app.py
```

Then type any email text and press Enter:

```
📨 Enter email text: You won a free iPhone! Click here to claim.
  🚫 Classification : SPAM
     Spam Probability : 96.43%
     Ham  Probability : 3.57%
```

### Option 3 — Use as a Python module

```python
from spam_classifier import predict

result = predict("Hi, are we still meeting at 3 PM today?")
print(result)
# {'label': 'HAM', 'confidence': '92.11%', 'spam_probability': '7.89%', 'ham_probability': '92.11%'}
```

### Option 4 — Run tests

```bash
pip install pytest
python -m pytest tests.py -v
```

---

## 🧠 How It Works

### 1. Data
The project uses a built-in dataset of 40 labeled emails (20 spam, 20 ham). No external download is needed.

### 2. TF-IDF Vectorization
Email text is converted into numerical features using **Term Frequency–Inverse Document Frequency (TF-IDF)**. Words that appear frequently in spam but rarely in ham get high scores, making them useful features.

### 3. Multinomial Naive Bayes
The classifier learns the probability of each word appearing in spam vs. ham emails. For a new email, it calculates which class is more probable given the words present.

### 4. Prediction
For each new email, the model outputs:
- A **label** (SPAM or HAM)
- **Spam probability** (0–100%)
- **Ham probability** (0–100%)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 70% |
| Spam Precision | 75% |
| Ham Precision | 67% |
| F1 Score (avg) | 0.70 |

> **Note:** Performance can be improved significantly by using a larger real-world dataset such as the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

---

## 🔮 Future Improvements

- Integrate a larger dataset (e.g., Enron Email Dataset)
- Try other classifiers: SVM, Random Forest, LSTM
- Build a web interface using Flask or Streamlit
- Add support for batch prediction via CSV upload
- Explore deep learning with word embeddings (Word2Vec, BERT)

---

## 👤 Author

**[Your Name]**  
AI/ML Capstone Project — VITyarthi Platform  
Submitted: 2025

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
