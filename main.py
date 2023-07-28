import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуем значения пикселей в диапазоне от 0 до 1
x_train, x_test = x_train / 255.0, x_test / 255.0


model = Sequential([
    Flatten(input_shape=(28, 28)),  # Преобразуем 28x28 изображение в одномерный вектор
    Dense(128, activation='relu'),    # Полносвязный слой с 128 нейронами и функцией активации ReLU
    Dense(10, activation='softmax')   # Выходной слой с 10 нейронами и функцией активации Softmax
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

