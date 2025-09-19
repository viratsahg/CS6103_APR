"""
------------------------------------------------------------
Student Name - Virat Prasad
Roll No. - 2511AI17
Subject - CS6103 | Advance Pattern Recognition
Submitted to - Dr. Chandranath Adak Sir
------------------------------------------------------------

Dataset: Iris (for classification using PCA + Logistic Regression)

------------------------------------------------------------
Install dependencies:
Command:  pip install numpy matplotlib seaborn scikit-learn
------------------------------------------------------------

Run this script directly:
    python iris_assignmt_code.py
------------------------------------------------------------

"""


# Import the required packages

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# PCA + Logistic Regression on Iris Dataset

def iris_pca_logreg(output_dir="output"):
    print("\n=== PCA + Logistic Regression on Iris Dataset ===")

    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic regression
    clf = LogisticRegression(max_iter=500, multi_class="ovr")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluation
    acc = clf.score(X_test, y_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    # Print to terminal
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(report)

    # Save results to file
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write("=== PCA + Logistic Regression on Iris Dataset ===\n\n")
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Plot PCA scatter
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=iris.target, palette="Set1")
    plt.title("Iris dataset after PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(os.path.join(output_dir, "pca_scatter.png"))
    plt.close()

    # Confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print(f"\nAll outputs saved in '{output_dir}/' folder.")



# main()

if __name__ == "__main__":
    iris_pca_logreg()
