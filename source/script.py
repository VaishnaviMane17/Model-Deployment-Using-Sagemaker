"""
This script is meant to run inside a SageMaker training job.
It receives S3 input data (automatically downloaded by SageMaker) via environment variables.
"""



import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.data_featureEngg import ThyroidFeatureEngineer
from source.DataPreprocessor import ThyroidDataPreprocessor


class RandomForestTrainer:
    def __init__(self, n_estimators=100, random_state=0):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, verbose=3, n_jobs=-1)

    def load_data(self, train_path, test_path, train_file='train-V-1.csv', test_file='test-V-1.csv'):
        print("[INFO] Loading training and test data...")
        train_df = pd.read_csv(os.path.join(train_path, train_file))
        test_df = pd.read_csv(os.path.join(test_path, test_file))

        features = list(train_df.columns)
        label = features.pop(-1)

        X_train = train_df[features]
        y_train = train_df[label]
        X_test = test_df[features]
        y_test = test_df[label]

        print("[INFO] Data loaded successfully.")
        return X_train, y_train, X_test, y_test

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        print("[INFO] Training Random Forest model...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("[RESULT] Test Accuracy:", acc)
        print("[RESULT] Classification Report:\n", report)
        return acc, report

    def save_model(self, model_dir):
        print("[INFO] Saving model to:", model_dir)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "model.joblib"))


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "data"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args = parser.parse_args()

    trainer = RandomForestTrainer(n_estimators=args.n_estimators, random_state=args.random_state)
    X_train, y_train, X_test, y_test = trainer.load_data(args.train, args.test, args.train_file, args.test_file)
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    trainer.save_model(args.model_dir)