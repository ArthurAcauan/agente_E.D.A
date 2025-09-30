# agent/eda_engine.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_csv(path_or_buffer, **kwargs):
    df = pd.read_csv(path_or_buffer, **kwargs)
    return df

def basic_summary(df):
    desc = df.describe(include='all').to_dict()
    dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
    nulls = df.isnull().sum().to_dict()
    shape = df.shape
    return {
        "shape": shape,
        "dtypes": dtypes,
        "nulls": nulls,
        "describe": desc
    }

def column_distribution(df, column, bins=30):
    series = df[column].dropna()
    counts, bin_edges = np.histogram(series, bins=bins)
    return {"counts": counts.tolist(), "bins": bin_edges.tolist()}

def outliers_iqr(df, column):
    s = df[column].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out = s[(s < lower) | (s > upper)]
    return {
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "lower_bound": float(lower), "upper_bound": float(upper),
        "outliers_count": int(out.shape[0]),
        "outliers_sample": out.head(10).tolist()
    }

def correlation_matrix(df, numeric_only=True):
    return df.corr(numeric_only=numeric_only).to_dict()

def kmeans_clusters(df, n_clusters=3, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(n_clusters, Xs.shape[0])
    if k <= 1:
        return {"message": "not enough numeric rows for clustering", "n_samples": Xs.shape[0]}
    km = KMeans(n_clusters=k, random_state=42).fit(Xs)
    labels = km.labels_.tolist()
    return {"n_clusters": k, "labels_sample": labels[:50]}

def top_frequent_values(df, column, n=10):
    return df[column].value_counts().head(n).to_dict()
