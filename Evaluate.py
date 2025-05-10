import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load the MNIST test dataset
def load_mnist_test_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize the data (scale pixel values to 0-1 range)
    x_test = x_test.astype("float32") / 255.0
    # Expand dimensions to fit model input shape (if required)
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test

# Load a pre-trained model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Plot misclassified samples
def plot_misclassified_samples(x_test, y_test, y_pred_classes, num_samples=24):
    misclassified_indices = np.where(y_test != y_pred_classes)[0]
    print(f"Number of misclassified samples: {len(misclassified_indices)}")
    
    # Select the first `num_samples` misclassified samples
    misclassified_indices = misclassified_indices[:num_samples]
    num_plots = len(misclassified_indices)
    
    # Define grid size (e.g., 3 columns)
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows dynamically
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()  # Flatten axes for easy iteration

    for i, idx in enumerate(misclassified_indices):
        ax = axes[i]
        ax.imshow(x_test[idx].squeeze(), cmap="gray")  # Remove single dimension for grayscale display
        ax.set_title(f"True: {y_test[idx]}\nPred: {y_pred_classes[idx]}")
        ax.axis("off")
    
    # Turn off unused axes
    for i in range(num_plots, len(axes)):
        axes[i].axis("off")
    

    plt.suptitle("Misclassified Samples", fontsize=16)
    plt.tight_layout()
    plt.savefig("Misclassified.png", dpi=300, bbox_inches="tight")
    # plt.show()

# Evaluate and plot performance
def evaluate_and_plot(model, x_test, y_test):
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Predict on the test data
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Custom plot for the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")  # `values_format="d"` ensures integers are displayed
    plt.title("Confusion Matrix", fontsize=16)
    
    # Save the plot
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'confusion_matrix.png'.")
    
    # Show the plot
    # plt.show()
    plot_misclassified_samples(x_test, y_test, y_pred_classes)

# Main script
if __name__ == "__main__":
    model_path = "models/RMSprop_mse_lr1e-03_num_channel64_num_conv3_num_dense3.h5"
      
    x_test, y_test = load_mnist_test_data()
    
    # Load the model
    model = load_model(model_path)
    
    # Evaluate and plot performance
    evaluate_and_plot(model, x_test, y_test)
