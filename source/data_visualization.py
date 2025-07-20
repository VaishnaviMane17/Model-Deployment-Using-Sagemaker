# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ThyroidVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def show_dtypes(self):
        print("\n🧾 Data Types:")
        print(self.df.dtypes)

    def show_null_summary(self):
        print("\n❗ Missing Values Summary:")
        print(self.df.isnull().sum())

    def show_statistics(self):
        print("\n📊 Descriptive Statistics:")
        print(self.df.describe(include='all'))

    def show_class_distribution(self):
        print("\n📌 Class Distribution:")
        class_counts = self.df['class'].value_counts()
        print(class_counts)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
        plt.title('Number of Patients in Each Class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def show_correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include='number')
        correlation_matrix = numeric_df.corr()

        plt.figure(figsize=(16, 14))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def visualize(self):
        self.show_dtypes()
        self.show_null_summary()
        self.show_statistics()
        self.show_class_distribution()
        self.show_correlation_matrix()
