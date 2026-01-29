"""
Скрипт для обучения модели диагностики гастрита.
"""

import os
import json
from datetime import datetime
import numpy as np
from dataset import GastritisDataset
from model import GastritisModel
from visualization import plot_training_history, plot_confusion_matrix, plot_feature_importance


def train_model(n_samples=1000, epochs=50, batch_size=32, test_size=0.2, val_size=0.1):
    """
    Полный цикл обучения модели.
    
    Args:
        n_samples: Количество образцов для генерации
        epochs: Количество эпох обучения
        batch_size: Размер батча
        test_size: Размер тестовой выборки
        val_size: Размер валидационной выборки
        
    Returns:
        Обученная модель и метрики
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ ДИАГНОСТИКИ ГАСТРИТА")
    print("=" * 60)
    
    # 1. Подготовка данных
    print("\n[1/5] Подготовка данных...")
    dataset = GastritisDataset()
    
    # Проверяем, существуют ли сохраненные данные
    data_path = 'data/gastritis_data.csv'
    if os.path.exists(data_path):
        print(f"Загрузка данных из {data_path}")
        df = dataset.load_data(data_path)
    else:
        print(f"Генерация {n_samples} синтетических образцов...")
        df = dataset.generate_synthetic_data(n_samples=n_samples)
        dataset.save_data(df)
    
    # Подготовка данных для обучения
    prepared_data = dataset.prepare_data(df, test_size=test_size, val_size=val_size)
    
    X_train = prepared_data['X_train']
    X_val = prepared_data['X_val']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_val = prepared_data['y_val']
    y_test = prepared_data['y_test']
    scaler = prepared_data['scaler']
    feature_names = prepared_data['feature_names']
    
    print(f"Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"Валидационная выборка: {X_val.shape[0]} образцов")
    print(f"Тестовая выборка: {X_test.shape[0]} образцов")
    
    # 2. Создание модели
    print("\n[2/5] Создание модели...")
    model = GastritisModel(input_shape=len(feature_names))
    model.create_model()
    print(model.get_model_summary())
    
    # 3. Обучение модели
    print("\n[3/5] Обучение модели...")
    print(f"Эпохи: {epochs}, Размер батча: {batch_size}")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=10
    )
    
    # 4. Оценка модели
    print("\n[4/5] Оценка модели...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nМетрики на тестовой выборке:")
    print(f"  Потеря (Loss): {metrics['loss']:.4f}")
    print(f"  Точность (Accuracy): {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # 5. Сохранение модели и скалера
    print("\n[5/5] Сохранение модели и скалера...")
    model.save_model()
    model.save_scaler(scaler)
    
    # 6. Визуализация результатов
    print("\n[6/6] Визуализация результатов...")
    
    # Создание директории для вывода
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # График истории обучения
    plot_training_history(history, save_path=f'{output_dir}/training_history.png')
    
    # Матрица ошибок
    y_pred, y_pred_proba = model.predict_batch(X_test)
    plot_confusion_matrix(y_test, y_pred, save_path=f'{output_dir}/confusion_matrix.png')
    
    # Важность признаков (на основе весов первого слоя)
    plot_feature_importance(
        model.model.layers[0].get_weights()[0],
        feature_names,
        save_path=f'{output_dir}/feature_importance.png'
    )
    
    # Сохранение метрик в JSON
    metrics_dict = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'epochs': epochs,
        'batch_size': batch_size,
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
        'metrics': {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        }
    }
    
    with open(f'{output_dir}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nМетрики сохранены в {output_dir}/metrics.json")
    
    # Тестовое предсказание
    print("\nТестовое предсказание:")
    test_sample = X_test[0:1]
    prediction, probability = model.predict(test_sample)
    print(f"  Симптомы: {test_sample[0]}")
    print(f"  Предсказание: {'Гастрит' if prediction == 1 else 'Нет гастрита'}")
    print(f"  Вероятность: {probability:.4f}")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    # Параметры обучения
    N_SAMPLES = 1000
    EPOCHS = 50
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Обучение модели
    model, metrics = train_model(
        n_samples=N_SAMPLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )
