import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Adadelta , Adagrad
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, concatenate, Input, Conv2D, Flatten, Lambda, MaxPooling2D
from utils import save_results_to_excel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import BinaryAccuracy

# Initialize accuracy metric
accuracy_metric = BinaryAccuracy()

(trainX, trainY), (testX, testY) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
a = int(random.random() * 5000)
plotting = False
if plotting ==True:
    # Assuming trainX contains the image data
    for i in range(9):
        # Define subplot for a 4x4 grid
        plt.subplot(3, 3, i + 1)
        # Plot raw pixel data
        plt.imshow(trainX[a+i], cmap='gray')

        # Show the figure
        # plt.tight_layout()
    plt.show()

# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainY = to_categorical(trainY)
testY = to_categorical(testY)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
# normalize to range 0-1
trainX /= 255.0
testX /= 255.0

# define cnn model
def Create_CNN_model(lr, optimizer, loss, num_channels, num_conv, num_dense):

  model = Sequential()
  model.add(Conv2D(num_channels, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))

  # Add increasing layers
  for i in range(num_conv):
    model.add(Conv2D(num_channels* (2 ** (i + 1)), (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())
  size = model.layers[-1].output_shape[1]
  print("Size is ",size)
  for i in range(num_dense):
    model.add(Dense(2000*(num_dense-i)/(num_dense+1), activation='relu'))
  model.add(Dense(10, activation='softmax'))
  # compile model
  model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=['accuracy'])
  model.summary()
  return model

# Define the parameter combinations
optimizers = [Adam, SGD, RMSprop]
loss_functions = [ 'mse', 'mae']
learning_rates = [0.001, 0.01, 0.1]
num_channels = [32, 64]
num_convs = [2, 3]
num_denses = [1, 3, 5]

# optimizers = [Adam]
# loss_functions = ['mse']
# learning_rates = [0.001]
# num_channels = [32]
# num_convs = [ 2 ]
# num_denses = [5]

results = []
# Test different combinations of optimizers, loss functions, and learning rates
for optimizer in optimizers:
    for loss in loss_functions:
        for lr in learning_rates:
            for num_channel in num_channels:
                for num_conv in num_convs:
                  for num_dense in num_denses:
                    try:
                        print(f'Testing {optimizer.__name__} with {loss} at learning rate {lr}')

                        # Create and compile the model
                        model = Create_CNN_model(lr, optimizer, loss, num_channel, num_conv, num_dense)

                        # Train the model
                        history = model.fit(trainX, trainY,
                                validation_data=(testX, testY),
                                epochs=20, batch_size=64)
                                # callbacks=[model_checkpoint])  # Adjust epochs as needed
                                # callbacks=[lr_scheduler, model_checkpoint])  # Adjust epochs as needed

                        # Evaluate the model
                        loss_value, mse_value = model.evaluate(trainX, trainY, verbose=0)
                        loss_value_test, mse_value_test = model.evaluate(testX, testY, verbose=0)
                        y_train_pred = model.predict(trainX)
                        y_test_pred = model.predict(testX)
                        
                        # Calculate metrics
                        
                        # Assume y_pred_train and y_train are your model predictions and true labels for training
                        # Similarly, y_pred_test and y_test are for testing
                        accuracy_metric.update_state(trainY, y_train_pred)
                        train_accuracy = accuracy_metric.result().numpy()  # Training accuracy

                        accuracy_metric.reset_states()  # Reset the metric state
                        accuracy_metric.update_state(testY, y_test_pred)
                        test_accuracy = accuracy_metric.result().numpy()  # Testing accuracy

                        mae_value_train = mean_absolute_error(trainY, y_train_pred)
                        mae_value_test = mean_absolute_error(testY, y_test_pred)
                        mse_value = mean_squared_error(trainY, y_train_pred)
                        mse_value_test = mean_squared_error(testY, y_test_pred)
                        model_name = f"{optimizer.__name__}_{loss}_lr{lr:.0e}_num_channel{num_channel}_num_conv{num_conv}_num_dense{num_dense}.h5"
                        model_path = os.path.join("models", model_name)
                        model.save(model_path)

                        # Find the epoch that corresponds to the best validation loss
                        best_epoch = np.argmin(history.history['val_loss']) + 1  # Adding 1 to account for 0-based index

                        print(f"The best model was saved at epoch: {best_epoch}")
                        trainable_params = sum(tf.size(variable).numpy() for variable in model.trainable_variables)
                        # Store the metrics in your results
                        results.append({
                            'optimizer': optimizer.__name__,
                            'loss_func': loss,
                            'learning_rate': lr,
                            'loss': loss_value,
                            'mse': mse_value,
                            'mae': mae_value_train,
                            'accuracy': train_accuracy,
                            'loss_test': loss_value_test,
                            'mse_test': mse_value_test,
                            'mae_test': mae_value_test,
                            'accuracy_test': test_accuracy,
                            'model_path': model_path,
                            'best_epoch': best_epoch,
                            'num_parameters': trainable_params
                        })
                    except Exception as e:
                        print(f"Error with configuration {optimizer.__name__}, {loss}, {lr}: {e}. Skipping...")
                        # Skip to the next iteration
                        continue


save_results_to_excel(results)
