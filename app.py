"""
Interactive Spam Email Classifier - Command Line Interface
Run this file to classify your own emails interactively.
"""

import os
import sys
from spam_classifier import load_data, preprocess, build_and_train, save_model, load_model, predict


def train_if_needed():
    """Train the model if not already saved."""
    if not os.path.exists("model/model.pkl"):
        print("No saved model found. Training now...")
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess(df)
        vectorizer, model = build_and_train(X_train, y_train)
        save_model(model, vectorizer)
        return model, vectorizer
    else:
        print("Loading saved model...")
        return load_model()


def interactive_mode(model, vectorizer):
    """Interactive prediction loop."""
    print("\n" + "="*60)
    print("  📧 Spam Email Classifier — Interactive Mode")
    print("="*60)
    print("  Type your email text and press Enter to classify.")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            email = input("📨 Enter email text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if email.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Thanks for using the Spam Classifier.")
            break

        if not email:
            print("  ⚠️  Please enter some text.\n")
            continue

        result = predict(email, model, vectorizer)
        icon = "🚫" if result["label"] == "SPAM" else "✅"
        print(f"\n  {icon} Classification : {result['label']}")
        print(f"     Spam Probability : {result['spam_probability']}")
        print(f"     Ham  Probability : {result['ham_probability']}\n")


if __name__ == "__main__":
    model, vectorizer = train_if_needed()
    interactive_mode(model, vectorizer)
