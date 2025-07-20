import pandas as pd
from typing import Optional

class ThyroidFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.replacement_values = {
            (1, 9, 'M'): 3.75, (1, 9, 'F'): 3.75,
            (10, 19, 'M'): 3.35, (10, 19, 'F'): 3.35,
            (20, 100, 'M'): 1.85, (20, 100, 'F'): 2.2
        }

        self.cols = ['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']

        self.target_mapping = {
            'A': 'Hyperthyroid', 'B': 'Hyperthyroid', 'C': 'Hyperthyroid', 'D': 'Hyperthyroid', 'AK': 'Hyperthyroid',
            'E': 'Hypothyroid', 'F': 'Hypothyroid', 'G': 'Hypothyroid', 'H': 'Hypothyroid', 'GK': 'Hypothyroid',
            'GI': 'Hypothyroid', 'FK': 'Hypothyroid', 'GKJ': 'Hypothyroid',
            'I': 'Binding Protein', 'J': 'Binding Protein', 'C|I': 'Binding Protein',
            'K': 'General Health', 'KJ': 'General Health', 'H|K': 'General Health',
            'M': 'Replacement Therapy', 'L': 'Replacement Therapy', 'N': 'Replacement Therapy',
            'MK': 'Replacement Therapy', 'MI': 'Replacement Therapy', 'LJ': 'Replacement Therapy',
            'O': 'Miscellaneous', 'P': 'Miscellaneous', 'Q': 'Miscellaneous', 'OI': 'Miscellaneous',
            'R': 'Miscellaneous', 'S': 'Miscellaneous', 'T': 'Miscellaneous', 'D|R': 'Miscellaneous',
            '-': 'No Condition'
        }

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    
    def drop_measured_flags(self) -> None:
        self.df.drop(columns=self.cols, inplace=True, errors='ignore')


    def fill_missing_sex(self) -> None:
        if 'sex' in self.df.columns:
            self.df['sex'].fillna(self.df['sex'].mode()[0], inplace=True)

    def replace_tbg(self) -> None:
        def replace(row):
            if pd.isnull(row['TBG']):
                age = row['age']
                sex = row['sex']
                for (low, high, g), value in self.replacement_values.items():
                    if low <= age <= high and sex == g:
                        return value
            return row['TBG']
        
        if 'TBG' in self.df.columns:
            self.df['TBG'] = self.df.apply(replace, axis=1)

    def map_target_column(self) -> None:
        if 'target' in self.df.columns:
            self.df['class'] = self.df['target'].map(self.target_mapping)
        if 'patient_id' in self.df.columns:
            self.df['Patient_ID'] = self.df.index + 1
            self.df.drop(columns=['target', 'patient_id'], inplace=True, errors='ignore')

    def drop_nan_if_no_condition(self) -> None:
        required_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'class']
        if all(col in self.df.columns for col in required_cols):
            condition = (
                (self.df['TSH'].isnull() | self.df['T3'].isnull() | self.df['TT4'].isnull() |
                 self.df['T4U'].isnull() | self.df['FTI'].isnull()) & 
                (self.df['class'] == 'No Condition')
            )
            self.df.drop(self.df[condition].index, inplace=True)

    def fill_remaining_nans(self):
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def clean(self) -> pd.DataFrame:
        self.drop_measured_flags()
        self.fill_missing_sex()
        self.replace_tbg()
        self.map_target_column()
        self.drop_nan_if_no_condition()
        self.fill_remaining_nans()
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def clean_and_save(self, output_path: Optional[str] = None) -> pd.DataFrame:
        df_cleaned = self.clean()
        df_cleaned.to_csv(output_path, index=False)
        print(f"[INFO] Saved cleaned data to data")
        return df_cleaned