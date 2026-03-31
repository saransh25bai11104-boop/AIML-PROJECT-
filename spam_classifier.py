"""
Spam Email Classifier using Machine Learning
A beginner-friendly AI/ML project using Naive Bayes and TF-IDF
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import pickle
import os


# ── Sample dataset (built-in so project works without downloads) ──────────────
SAMPLE_DATA = {
    "text": [
        # SPAM examples
        "Congratulations! You have won a $1,000 gift card. Click here to claim now!",
        "FREE MONEY! Earn $500 per day working from home. No experience needed!",
        "URGENT: Your account has been compromised. Verify your details immediately.",
        "You are selected for a special prize! Send your bank details to receive $10,000.",
        "Buy cheap medications online! Viagra, Cialis — no prescription needed!",
        "Click here to enlarge and impress! Limited time offer!",
        "Nigerian Prince needs your help. Share profits of $50 million with you.",
        "Win a free iPhone 15! Just complete this survey. Offer expires TODAY!",
        "Lose 30 pounds in 30 days! Miracle weight loss pill — guaranteed results!",
        "Hot singles in your area want to meet you! Sign up for FREE!",
        "Your PayPal account is suspended. Confirm identity at this link urgently.",
        "You have been pre-approved for a $50,000 loan! No credit check required!",
        "Make millions online with this secret method banks don't want you to know!",
        "FREE vacation package for you and your family! Click to claim now!",
        "WINNER! You are today's lucky visitor. Collect your prize before midnight!",
        "Cheap Rolex watches! 90% off luxury brands — limited stock available!",
        "INVESTMENT OPPORTUNITY: Double your Bitcoin in 24 hours guaranteed!",
        "Earn easy cash from home. Our members make $3000/week. Join for free!",
        "Your computer is infected with viruses! Call this number immediately!",
        "Special offer for loyal customers: 80% off all products this week only!",
        # HAM (legitimate) examples
        "Hi Sarah, are we still on for lunch tomorrow at 12:30?",
        "Please find attached the quarterly report for your review and feedback.",
        "The team meeting has been rescheduled to Thursday at 2 PM. Please update your calendars.",
        "Thank you for your order. Your package will arrive within 3-5 business days.",
        "Reminder: Your dentist appointment is on Friday, June 14th at 10:00 AM.",
        "Could you please review the pull request I submitted this morning? Thanks!",
        "Happy Birthday! Hope you have a wonderful day filled with joy.",
        "I will be working from home tomorrow due to a doctor's appointment.",
        "The project deadline has been extended to next Friday. Please plan accordingly.",
        "Can you send me the latest version of the presentation file? Need it for Monday.",
        "Your subscription has been renewed. Invoice is attached for your records.",
        "Great job on the presentation today! The client was very impressed.",
        "Library book reminder: 'Python for Beginners' is due back on June 20th.",
        "Weekly standup notes attached. Action items highlighted in yellow.",
        "Your flight booking confirmation: Mumbai to Delhi on June 18, seat 14A.",
        "We are excited to announce our annual team picnic on July 4th at Central Park.",
        "Please submit your timesheet by end of day Friday. HR team.",
        "Your internet bill of Rs. 999 is due on June 25. Pay online or at any branch.",
        "The server maintenance window is scheduled for Saturday 2 AM - 4 AM.",
        "Mom called. She wants you to call her back when you get a chance.",
    ],
    "label": (["spam"] * 20) + (["ham"] * 20)
}


def load_data() -> pd.DataFrame:
    """Load or generate the email dataset."""
    df = pd.DataFrame(SAMPLE_DATA)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    print(f"Dataset loaded: {len(df)} emails ({df['label'].value_counts()['spam']} spam, {df['label'].value_counts()['ham']} ham)")
    return df


def preprocess(df: pd.DataFrame):
    """Convert labels to binary and split into train/test sets."""
    df["label_num"] = df["label"].map({"spam": 1, "ham": 0})
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"], test_size=0.25, random_state=42, stratify=df["label_num"]
    )
    return X_train, X_test, y_train, y_test


def build_and_train(X_train, y_train):
    """Build TF-IDF vectorizer + Naive Bayes classifier pipeline."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


def evaluate(model, vectorizer, X_test, y_test):
    """Evaluate model and print metrics."""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Model Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])}")

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':10} Predicted Ham  Predicted Spam")
    print(f"  Actual Ham   {cm[0][0]:^13} {cm[0][1]:^14}")
    print(f"  Actual Spam  {cm[1][0]:^13} {cm[1][1]:^14}")
    print(f"{'='*50}\n")
    return acc


def save_model(model, vectorizer, path="model"):
    """Save trained model and vectorizer to disk."""
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{path}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Model saved to ./{path}/")


def load_model(path="model"):
    """Load saved model and vectorizer."""
    with open(f"{path}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{path}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None, model_path="model") -> dict:
    """
    Predict whether an email is spam or ham.

    Args:
        text: Email content string
        model: Pre-loaded model (optional)
        vectorizer: Pre-loaded vectorizer (optional)
        model_path: Path to saved model if not pre-loaded

    Returns:
        dict with 'label', 'confidence', and 'probability'
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model(model_path)

    tfidf = vectorizer.transform([text])
    pred = model.predict(tfidf)[0]
    prob = model.predict_proba(tfidf)[0]

    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[pred] * 100

    return {
        "label": label,
        "confidence": f"{confidence:.2f}%",
        "spam_probability": f"{prob[1]*100:.2f}%",
        "ham_probability": f"{prob[0]*100:.2f}%"
    }


def demo_predictions(model, vectorizer):
    """Run demo predictions on new emails."""
    test_emails = [
        "You won a free trip to Goa! Claim your prize by clicking this link now!",
        "Hi, just checking if you received my last email about the project update.",
        "URGENT: Transfer $500 now to unlock your lottery winnings of $100,000!",
        "The monthly team report is attached. Please review before Thursday's meeting.",
        "Congratulations, your resume has been shortlisted for the Software Engineer role.",
    ]

    print("Demo Predictions on New Emails:")
    print("="*60)
    for email in test_emails:
        result = predict(email, model, vectorizer)
        print(f"\nEmail : {email[:60]}...")
        print(f"Result: {result['label']} (Spam: {result['spam_probability']}, Ham: {result['ham_probability']})")
    print("="*60)


if __name__ == "__main__":
    print("\n🚀 Spam Email Classifier — AI/ML Capstone Project")
    print("=" * 50)

    # Step 1: Load data
    df = load_data()

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Step 3: Train
    print("\nTraining Naive Bayes classifier with TF-IDF features...")
    vectorizer, model = build_and_train(X_train, y_train)
    print("Training complete!")

    # Step 4: Evaluate
    accuracy = evaluate(model, vectorizer, X_test, y_test)

    # Step 5: Save
    save_model(model, vectorizer)

    # Step 6: Demo
    demo_predictions(model, vectorizer)

    print("✅ Project complete! Model saved. You can now run app.py for interactive predictions.\n")
