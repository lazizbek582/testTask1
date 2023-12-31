# Разработка нейронной сети для классификации изображений MNIST

Данный репозиторий содержит код для простой нейронной сети, которая выполняет классификацию рукописных изображений цифр от 0 до 9 из набора данных MNIST, используя библиотеку TensorFlow.

## Подготовка данных

Для обучения модели используется набор данных MNIST, который предоставляется библиотекой TensorFlow. Изображения подготавливаются путем нормализации, чтобы значения пикселей находились в диапазоне от 0 до 1.

## Архитектура модели

Модель состоит из трех слоев:
1. Первый слой (входной слой) - Flatten: Преобразует изображение 28x28 в одномерный вектор для подачи данных в следующий слой.
2. Второй слой - Dense: Полносвязный слой с 128 нейронами и функцией активации ReLU. Этот слой выполняет внутреннюю обработку данных и извлекает признаки из входных данных.
3. Третий слой - Dense (выходной слой): Выходной слой с 10 нейронами и функцией активации Softmax. Softmax преобразует выходные значения в вероятности, что позволяет определить вероятность принадлежности изображения к каждому из классов (цифры от 0 до 9).

## Обучение и оценка производительности

После создания модели, она компилируется с оптимизатором "adam", функцией потерь "sparse_categorical_crossentropy" и метрикой "accuracy" для оценки производительности.

Модель обучается на обучающем наборе данных с использованием функции `model.fit()`. В данном примере обучение проходит в течение 100 эпох с размером пакета 32.

## Точность модели

По результатам обучения, модель достигает точности около 99.9% на валидационных данных.

## Установка

Для использования кода, необходимо установить библиотеку TensorFlow:

```bash
pip install tensorflow
