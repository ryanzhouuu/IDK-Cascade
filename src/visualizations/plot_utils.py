import matplotlib.pyplot as plt
import numpy as np

def plot_cascade_performance(metrics_history):
    """
    Plot the cascade performance metrics over time

    Args:
        metrics_history (dict): Dictionary containing lists of metrics
    """

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['accuracy'], label='Accuracy')
    plt.plot(metrics_history['coverage'], label='Coverage')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Accuracy and Coverage')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['idk_rate'], label='IDK Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('IDK Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confidence_distribution(confidences, is_idk):
    """
    Plot the distribution of confidences scores

    Args:
        confidences (np.array): Array of confidence scores
        is_idk (np.array): Boolean array indicating IDK cases
    """

    plt.figure(figsize=(10, 6))

    plt.hist(confidences[~is_idk], bins=50, alpha=0.5, label='Accepted')
    plt.hist(confidences[is_idk], bins=50, alpha=0.5, label='IDK')

    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    plt.legend()

    plt.show()