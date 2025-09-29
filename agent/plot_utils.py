# agent/plot_utils.py
import matplotlib.pyplot as plt
import io
import base64

def save_histogram(df, column, bins=30, figsize=(8,4), show=False):
    plt.figure(figsize=figsize)
    df[column].dropna().hist(bins=bins)
    plt.title(f"Histograma: {column}")
    plt.xlabel(column)
    plt.ylabel("Frequência")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def save_scatter(df, x, y, figsize=(6,4)):
    plt.figure(figsize=figsize)
    plt.scatter(df[x], df[y], s=5, alpha=0.6)
    plt.xlabel(x); plt.ylabel(y)
    plt.title(f"Scatter: {x} vs {y}")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def buf_to_base64_png(buf):
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"
