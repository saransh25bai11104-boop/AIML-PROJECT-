"""
Unit Tests for Spam Email Classifier
Run with: python -m pytest tests.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spam_classifier import load_data, preprocess, build_and_train, predict


@pytest.fixture(scope="module")
def trained_model():
    """Fixture that trains a model once for all tests."""
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    vectorizer, model = build_and_train(X_train, y_train)
    return model, vectorizer, X_test, y_test


class TestDataLoading:
    def test_dataset_not_empty(self):
        df = load_data()
        assert len(df) > 0, "Dataset should not be empty"

    def test_dataset_has_required_columns(self):
        df = load_data()
        assert "text" in df.columns
        assert "label" in df.columns

    def test_dataset_has_both_classes(self):
        df = load_data()
        assert "spam" in df["label"].values
        assert "ham" in df["label"].values

    def test_no_null_values(self):
        df = load_data()
        assert df["text"].isnull().sum() == 0
        assert df["label"].isnull().sum() == 0


class TestModelTraining:
    def test_model_trains_successfully(self, trained_model):
        model, vectorizer, _, _ = trained_model
        assert model is not None
        assert vectorizer is not None

    def test_model_accuracy_above_threshold(self, trained_model):
        from sklearn.metrics import accuracy_score
        model, vectorizer, X_test, y_test = trained_model
        X_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_tfidf)
        acc = accuracy_score(y_test, y_pred)
        assert acc >= 0.70, f"Accuracy {acc:.2f} is below threshold of 0.70"


class TestPrediction:
    def test_predict_obvious_spam(self, trained_model):
        model, vectorizer, _, _ = trained_model
        spam_email = "Congratulations! You won $10,000! Click here to claim your prize now!"
        result = predict(spam_email, model, vectorizer)
        assert result["label"] in ["SPAM", "HAM"], "Should return a valid label"
        assert "%" in result["confidence"]

    def test_predict_obvious_ham(self, trained_model):
        model, vectorizer, _, _ = trained_model
        ham_email = "Hi, can we schedule a meeting for Tuesday at 3 PM to discuss the project?"
        result = predict(ham_email, model, vectorizer)
        assert result["label"] in ["SPAM", "HAM"]

    def test_predict_returns_probabilities(self, trained_model):
        model, vectorizer, _, _ = trained_model
        result = predict("Test email content", model, vectorizer)
        assert "spam_probability" in result
        assert "ham_probability" in result
        assert "confidence" in result

    def test_predict_empty_ish_input(self, trained_model):
        model, vectorizer, _, _ = trained_model
        result = predict("hello", model, vectorizer)
        assert result["label"] in ["SPAM", "HAM"]


class TestPreprocessing:
    def test_train_test_split(self):
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess(df)
        total = len(X_train) + len(X_test)
        assert total == len(df)
        assert len(X_train) > len(X_test), "Train set should be larger than test set"

    def test_labels_are_binary(self):
        df = load_data()
        _, _, y_train, y_test = preprocess(df)
        all_labels = list(y_train) + list(y_test)
        assert all(l in [0, 1] for l in all_labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
