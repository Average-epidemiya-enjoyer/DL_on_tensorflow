import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from main import *

X = np.arange(-100, 100, 4)

y = np.arange(-90, 110, 4)
y = X + 10

X_train = X[:int(0.8 * len(X))]
y_train = y[:int(0.8 * len(X))]

X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(X)):]

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.scatter(X_test, y_test, c='g', label='Testing data')
plt.legend()
plt.show()

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

print(model.summary())
model.fit(X_train, y_train, epochs=100, verbose=0)
print(model.summary())

y_preds = model.predict(X_test)


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()


plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)
