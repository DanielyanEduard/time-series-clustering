import matplotlib.pyplot as plt
import numpy as np
# Function to visualize time series data by true labels
def plot_time_series_by_class(X, y, title):
    plt.figure(figsize=(12, 8))
    classes = np.unique(y)

    for i, cls in enumerate(classes):
        plt.subplot(len(classes), 1, i + 1)

        # Plot each time series in this class
        for ts in X[y == cls]:
            plt.plot(ts.ravel(), 'k-', alpha=0.2)

        # Plot the average time series for this class
        mean_ts = np.mean(X[y == cls], axis=0)
        plt.plot(mean_ts.ravel(), 'r-', linewidth=2)

        plt.title(f'Class {cls}')
        plt.tight_layout()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return plt


# Function to visualize clustering results
def plot_clustering_results(X, labels, title):
    plt.figure(figsize=(12, 8))
    clusters = np.unique(labels)

    for i, cls in enumerate(clusters):
        plt.subplot(len(clusters), 1, i + 1)

        # Plot each time series in this cluster
        for ts in X[labels == cls]:
            plt.plot(ts.ravel(), 'k-', alpha=0.2)

        # Plot the average time series for this cluster
        if np.sum(labels == cls) > 0:  # Check if cluster has members
            mean_ts = np.mean(X[labels == cls], axis=0)
            plt.plot(mean_ts.ravel(), 'r-', linewidth=2)

        plt.title(f'Cluster {cls}')
        plt.tight_layout()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return plt

# Create confusion matrices to visualize the relationship between
# true classes and predicted clusters
def plot_confusion_matrix(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true-1, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Cluster')
    plt.tight_layout()
    return plt