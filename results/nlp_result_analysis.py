import numpy as np


if __name__ == "__main__":
    # Load the numpy result
    result = np.load("transformers_nlp_sentiment_analysis.npy")

    # Get the individual result
    train_loss_array = result[0]
    val_loss_array = result[1]
    train_accuracy_array = result[2]
    val_accuracy_array = result[3]