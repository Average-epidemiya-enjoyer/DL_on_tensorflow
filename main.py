import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

plt.scatter(X, y)
plt.show()

input_shape = X[0].shape

output_shape = y[0].shape

tf.random.set_seed(42)

# Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,  # СКО
              optimizer=tf.keras.optimizers.SGD(),  # Тут используем стохастический градиентый спуск
              metrics=["mae"])

model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

print(model.predict([17.0]))
