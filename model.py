"""
Модуль для создания и обучения модели машинного обучения для диагностики гастрита.
Использует нейронную сеть на основе TensorFlow/Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from datetime import datetime


class GastritisModel:
    """Класс для создания, обучения и использования модели диагностики гастрита."""
    
    def __init__(self, input_shape=10, model_path='models/gastritis_model'):
        """
        Инициализация модели.
        
        Args:
            input_shape: Количество входных признаков (симптомов)
            model_path: Путь для сохранения модели
        """
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        self.history = None
        
    def create_model(self):
        """
        Создание архитектуры нейронной сети.
        
        Returns:
            Скомпилированная модель Keras
        """
        model = keras.Sequential([
            # Входной слой
            layers.Input(shape=(self.input_shape,)),
            
            # Первый скрытый слой
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Второй скрытый слой
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Третий скрытый слой
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Выходной слой (бинарная классификация)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Компиляция модели
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """
        Получение сводки архитектуры модели.
        
        Returns:
            Строка со сводкой модели
        """
        if self.model is None:
            self.create_model()
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()
        stream.close()
        return summary_str
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, early_stopping_patience=10):
        """
        Обучение модели.
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные
            y_val: Валидационные метки
            epochs: Количество эпох обучения
            batch_size: Размер батча
            early_stopping_patience: Патience для early stopping
            
        Returns:
            История обучения
        """
        if self.model is None:
            self.create_model()
        
        # Callbacks для обучения
        callbacks = [
            # Early stopping для предотвращения переобучения
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Сохранение лучшей модели
            keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Обучение модели
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Оценка модели на тестовых данных.
        
        Args:
            X_test: Тестовые данные
            y_test: Тестовые метки
            
        Returns:
            Словарь с метриками оценки
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Оценка модели
        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Вычисление F1-score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        return metrics
    
    def predict(self, X, threshold=0.5):
        """
        Предсказание диагноза.
        
        Args:
            X: Входные данные (симптомы)
            threshold: Порог для классификации
            
        Returns:
            Кортеж (предсказанный класс, вероятность)
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Предсказание вероятности
        probability = self.model.predict(X, verbose=0)[0][0]
        
        # Классификация по порогу
        prediction = 1 if probability >= threshold else 0
        
        return prediction, probability
    
    def predict_batch(self, X, threshold=0.5):
        """
        Предсказание диагноза для нескольких образцов.
        
        Args:
            X: Входные данные (симптомы)
            threshold: Порог для классификации
            
        Returns:
            Кортеж (предсказанные классы, вероятности)
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Предсказание вероятностей
        probabilities = self.model.predict(X, verbose=0).flatten()
        
        # Классификация по порогу
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def save_model(self, filepath=None):
        """
        Сохранение модели.
        
        Args:
            filepath: Путь для сохранения (если None, используется model_path)
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала вызовите метод create_model().")
        
        if filepath is None:
            filepath = f'{self.model_path}.keras'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath=None):
        """
        Загрузка модели.
        
        Args:
            filepath: Путь к файлу модели (если None, используется model_path)
        """
        if filepath is None:
            filepath = f'{self.model_path}.keras'
        
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Модель загружена из {filepath}")
        else:
            raise FileNotFoundError(f"Файл модели {filepath} не найден.")
    
    def save_scaler(self, scaler, filepath='models/scaler.pkl'):
        """
        Сохранение скалера для нормализации данных.
        
        Args:
            scaler: Объект скалера
            filepath: Путь для сохранения
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(scaler, filepath)
        print(f"Скалер сохранен в {filepath}")
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        """
        Загрузка скалера.
        
        Args:
            filepath: Путь к файлу скалера
            
        Returns:
            Загруженный скалер
        """
        if os.path.exists(filepath):
            scaler = joblib.load(filepath)
            print(f"Скалер загружен из {filepath}")
            return scaler
        else:
            raise FileNotFoundError(f"Файл скалера {filepath} не найден.")


if __name__ == "__main__":
    # Пример использования
    print("Создание модели...")
    model = GastritisModel(input_shape=10)
    
    # Создание модели
    model.create_model()
    
    # Вывод архитектуры
    print("\nАрхитектура модели:")
    print(model.get_model_summary())
    
    # Генерация тестовых данных
    np.random.seed(42)
    X_train = np.random.randn(800, 10)
    y_train = np.random.randint(0, 2, 800)
    X_val = np.random.randn(100, 10)
    y_val = np.random.randint(0, 2, 100)
    
    print("\nОбучение модели...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=5)
    
    print("\nОценка модели...")
    X_test = np.random.randn(100, 10)
    y_test = np.random.randint(0, 2, 100)
    metrics = model.evaluate(X_test, y_test)
    print(f"Метрики: {metrics}")
    
    print("\nПредсказание...")
    X_new = np.random.randn(1, 10)
    prediction, probability = model.predict(X_new)
    print(f"Предсказание: {prediction}, Вероятность: {probability:.4f}")
