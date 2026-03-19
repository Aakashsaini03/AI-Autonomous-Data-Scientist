import pandas as pd

def load_data(file):
    return pd.read_csv(file)

def data_summary(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

def clean_data(df):
    df = df.copy()  # 🔥 important

    # Numeric columns
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())

    # Categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df