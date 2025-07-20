import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN


class ThyroidDataPreprocessor:
    def __init__(self, data: pd.DataFrame, target_col: str = 'class'):
        self.data = data
        self.target_col = target_col

    def split_and_resample(self, test_size=0.2, random_state=42):
        print("[INFO] Splitting data...")
        X = self.data.drop([self.target_col], axis=1)
        y = self.data[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Encode categorical features
        X_train_encoded = pd.get_dummies(X_train)
        X_test_encoded = pd.get_dummies(X_test)

        # Align test with train in case dummy columns mismatch
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

        smoteenn = SMOTEENN(random_state=random_state)
        X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_encoded, y_train)

        trainX = pd.DataFrame(X_train_resampled)
        trainX['label'] = y_train_resampled

        testX = pd.DataFrame(X_test)
        testX['label'] = y_test

        print(f"Train set shape after SMOTEENN: {trainX.shape}, Test set shape: {testX.shape}")

        return trainX, testX

    def save_to_csv(self, train_df: pd.DataFrame, test_df: pd.DataFrame, train_file='train-V-1.csv', test_file='test-V-1.csv'):
        print(f"[INFO] Saving train to {train_file} and test to {test_file}")
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)






