import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_histogram(data, title="", xlabel="", ylabel="Frequency", bins=30):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_scatter(x, y, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_bar_counts(categories, title="", xlabel="", ylabel="Count", rotation=45):
    unique_vals, counts = np.unique(categories, return_counts=True)
    plt.figure(figsize=(7, 4))
    plt.bar(unique_vals, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_pie(categories, title=""):
    unique_vals, counts = np.unique(categories, return_counts=True)
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=unique_vals, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_line(x, y, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(X, feature_names=None, title="Correlation Heatmap"):
    corr = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(8, 6))
    if feature_names is None:
        sns.heatmap(corr, annot=False)
    else:
        sns.heatmap(corr, annot=False, xticklabels=feature_names, yticklabels=feature_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_experience_bar(labels, values, title="Tỷ lệ muốn đổi việc theo kinh nghiệm"):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xlabel("Số năm kinh nghiệm")
    plt.ylabel("Tỷ lệ muốn đổi việc (target=1)")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()