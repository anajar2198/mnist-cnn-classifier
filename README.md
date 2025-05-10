
# 🧠 MNIST Handwritten Digit Classification using Configurable CNNs

This project implements a flexible deep learning framework using TensorFlow and Keras to classify MNIST digits. We examine how different model architectures and hyperparameter combinations affect performance.

---

## 📦 Dataset: MNIST

- **Source**: `tensorflow.keras.datasets.mnist`
- **Train Set**: 60,000 grayscale images (28x28)
- **Test Set**: 10,000 images
- **Classes**: 10 (digits 0–9)

---

## 🧰 Environment Setup

```bash
conda create -n Mnist python=3.8.0
conda activate Mnist
conda install -c conda-forge tensorflow
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

---

## 🔍 Data Preparation

- Reshape images to shape (28, 28, 1)
- Normalize pixel values to [0, 1]
- Labels converted to one-hot vectors using `to_categorical`

---

## 🧠 Model Architecture: Dynamic CNN

We define a flexible `Create_CNN_model()` function with arguments:

- `lr`: learning rate
- `optimizer`: optimizer function
- `loss`: loss function
- `num_channels`: filters in first Conv2D layer
- `num_conv`: number of Conv2D+Pool blocks
- `num_dense`: number of Dense layers

### 🧱 Structure

- Conv2D → ReLU → MaxPooling2D (× N)
- Flatten
- Dense layers (× M)
- Output layer: Dense(10, softmax)

Example layer scaling: each additional Conv2D doubles the number of filters.

---

## ⚙️ Hyperparameter Search

We evaluate combinations of the following:

| Parameter       | Values                            |
|----------------|------------------------------------|
| Optimizers      | `Adam`, `SGD`, `RMSprop`           |
| Loss functions  | `mse`, `mae`                       |
| Learning rates  | `0.001`, `0.01`, `0.1`             |
| Channels        | `32`, `64`                         |
| Conv2D layers   | `2`, `3`                           |
| Dense layers    | `1`, `3`, `5`                      |

Each configuration is trained for 20 epochs with batch size 64.

---

## 📊 Evaluation Metrics

After each training:

- `loss`, `mse`, `mae`, `accuracy` (train/test)
- Best validation epoch
- Number of parameters
- Model saved to `models/` folder
- Results logged in `results.xlsx`

---

## 🏆 Best Models

| Optimizer | Loss | LR    | Conv | Dense | Test Accuracy |
|----------:|-----:|------:|-----:|------:|---------------:|
| RMSprop   | mse  | 0.001 | 3    | 3     | **99.89%**     |
| SGD       | mse  | 0.1   | 3    | 3     | 99.76%         |
| Adam      | mse  | 0.01  | 3    | 3     | 89.99%         |

---

## 🔬 Visualizations

We plot:
- Confusion matrix of best model
- Misclassified digits
- Accuracy vs optimizer/learning rate
- Accuracy vs network size

---

## 📁 Project Files

```
mnist-cnn/
├── model.py             # defines Create_CNN_model + training loops
├── utils.py             # helper: save_results_to_excel()
├── results.xlsx         # final evaluation results
├── README.md
```

---

## ▶️ How to Run

```bash
python model.py
```

This script:
- Loops through all configurations
- Trains each model
- Evaluates and saves metrics
- Logs results in `results.xlsx`

---

## 📬 Author

**Abolfazl Najar**  
📧 anajar2198@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/abolfazl-najar)

---

## 📜 License

MIT License — free to use and modify.

---

> *If you use this project in your work, please cite or star the repository.*
