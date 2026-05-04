import numpy as np
import pandas as pd


API_ATTACKS = [
    'SQL Injection', 'DDoS', 'Brute Force Attack',
    'Cross-site Scripting (XSS)', 'Privilege Escalation', 'Man-in-the-Middle'
]

SEVERITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temporal features from Date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Damage-based features
    df['Log_Damage'] = np.log1p(df['Damage_Estimate(USD)'])
    df['Damage_Per_System'] = df['Damage_Estimate(USD)'] / (df['Affected_Systems'] + 1)
    df['Log_Damage_Per_System'] = np.log1p(df['Damage_Per_System'])
    high_damage_threshold = df['Damage_Estimate(USD)'].quantile(0.75)
    df['High_Damage'] = (df['Damage_Estimate(USD)'] >= high_damage_threshold).astype(int)

    # Severity as ordinal numeric
    df['Severity_Numeric'] = df['Severity'].map(SEVERITY_MAP).fillna(1)

    # Binary label: is this an API-specific attack?
    df['Is_API_Attack'] = df['Attack_Type'].isin(API_ATTACKS).astype(int)

    # Risk scores derived from group averages (computed on full dataset)
    industry_risk = df.groupby('Target_Industry')['Severity_Numeric'].mean()
    df['Industry_Risk_Score'] = df['Target_Industry'].map(industry_risk)

    location_risk = df.groupby('Location')['Severity_Numeric'].mean()
    df['Location_Risk_Score'] = df['Location'].map(location_risk)

    # Attack frequency per location / industry
    location_freq = df.groupby('Location').size() / len(df)
    df['Location_Attack_Frequency'] = df['Location'].map(location_freq)

    industry_freq = df.groupby('Target_Industry').size() / len(df)
    df['Industry_Attack_Frequency'] = df['Target_Industry'].map(industry_freq)

    return df


def get_feature_columns() -> tuple[list, list]:
    """Return (categorical_cols, numerical_cols) used for modelling."""
    categorical_cols = ['Target_Industry', 'Location', 'Severity']
    numerical_cols = [
        'Year', 'Month', 'DayOfWeek', 'Quarter', 'IsWeekend',
        'Log_Damage', 'Affected_Systems', 'Log_Damage_Per_System',
        'Industry_Risk_Score', 'Location_Risk_Score',
        'Industry_Attack_Frequency', 'Location_Attack_Frequency',
        'High_Damage', 'Severity_Numeric'
    ]
    return categorical_cols, numerical_cols
