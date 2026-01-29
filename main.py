"""
Главный файл приложения для диагностики гастрита с использованием искусственного интеллекта.
Система предназначена для предварительной установки диагноза на основе симптомов пациента.

ВНИМАНИЕ: Данная система является только вспомогательным инструментом и не заменяет
консультацию квалифицированного врача. Все диагнозы должны быть подтверждены специалистом.
"""

import os
import sys
from datetime import datetime
from dataset import GastritisDataset
from model import GastritisModel
from predict import GastritisPredictor
from visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_symptoms_comparison,
    plot_prediction_probability,
    plot_metrics_comparison
)


def print_header():
    """Вывод заголовка приложения."""
    print("\n" + "=" * 70)
    print(" " * 15 + "СИСТЕМА ДИАГНОСТИКИ ГАСТРИТА")
    print(" " * 10 + "С ИСПОЛЬЗОВАНИЕМ ИСКУССТВЕННОГО ИНТЕЛЛЕКТА")
    print("=" * 70)
    print("\nВНИМАНИЕ: Данная система является только вспомогательным инструментом")
    print("и не заменяет консультацию квалифицированного врача.")
    print("Все диагнозы должны быть подтверждены специалистом.")
    print("=" * 70 + "\n")


def print_menu():
    """Вывод главного меню."""
    print("\n" + "-" * 70)
    print("ГЛАВНОЕ МЕНЮ")
    print("-" * 70)
    print("  1. Обучить модель")
    print("  2. Диагностика пациента")
    print("  3. Сравнить симптомы двух пациентов")
    print("  4. Просмотреть информацию о модели")
    print("  5. Показать примеры визуализации")
    print("  6. Выход")
    print("-" * 70)


def train_model_menu():
    """Меню обучения модели."""
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)
    
    # Параметры обучения
    print("\nПараметры обучения (по умолчанию):")
    print("  Количество образцов: 1000")
    print("  Количество эпох: 50")
    print("  Размер батча: 32")
    print("  Размер тестовой выборки: 20%")
    print("  Размер валидационной выборки: 10%")
    
    use_default = input("\nИспользовать параметры по умолчанию? (д/н): ").lower()
    
    if use_default == 'д' or use_default == 'y' or use_default == 'yes':
        n_samples = 1000
        epochs = 50
        batch_size = 32
        test_size = 0.2
        val_size = 0.1
    else:
        try:
            n_samples = int(input("Количество образцов: "))
            epochs = int(input("Количество эпох: "))
            batch_size = int(input("Размер батча: "))
            test_size = float(input("Размер тестовой выборки (0-1): "))
            val_size = float(input("Размер валидационной выборки (0-1): "))
        except ValueError:
            print("\nОшибка ввода. Используются параметры по умолчанию.")
            n_samples = 1000
            epochs = 50
            batch_size = 32
            test_size = 0.2
            val_size = 0.1
    
    # Подготовка данных
    print("\n[1/5] Подготовка данных...")
    dataset = GastritisDataset()
    
    data_path = 'data/gastritis_data.csv'
    if os.path.exists(data_path):
        print(f"Загрузка данных из {data_path}")
        df = dataset.load_data(data_path)
    else:
        print(f"Генерация {n_samples} синтетических образцов...")
        df = dataset.generate_synthetic_data(n_samples=n_samples)
        dataset.save_data(df)
    
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
    
    # Создание модели
    print("\n[2/5] Создание модели...")
    model = GastritisModel(input_shape=len(feature_names))
    model.create_model()
    print(model.get_model_summary())
    
    # Обучение модели
    print("\n[3/5] Обучение модели...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=10
    )
    
    # Оценка модели
    print("\n[4/5] Оценка модели...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nМетрики на тестовой выборке:")
    print(f"  Потеря (Loss): {metrics['loss']:.4f}")
    print(f"  Точность (Accuracy): {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # Сохранение модели
    print("\n[5/5] Сохранение модели и скалера...")
    model.save_model()
    model.save_scaler(scaler)
    
    # Визуализация
    print("\n[6/6] Визуализация результатов...")
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_training_history(history, save_path=f'{output_dir}/training_history.png')
    
    y_pred, y_pred_proba = model.predict_batch(X_test)
    plot_confusion_matrix(y_test, y_pred, save_path=f'{output_dir}/confusion_matrix.png')
    
    plot_feature_importance(
        model.model.layers[0].get_weights()[0],
        feature_names,
        save_path=f'{output_dir}/feature_importance.png'
    )
    
    plot_metrics_comparison(metrics, save_path=f'{output_dir}/metrics_comparison.png')
    
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 70)
    
    input("\nНажмите Enter для продолжения...")


def diagnose_patient_menu():
    """Меню диагностики пациента."""
    print("\n" + "=" * 70)
    print("ДИАГНОСТИКА ПАЦИЕНТА")
    print("=" * 70)
    
    # Проверка наличия модели
    if not os.path.exists('models/gastritis_model.keras'):
        print("\nОШИБКА: Обученная модель не найдена!")
        print("Пожалуйста, сначала обучите модель (пункт 1 в главном меню).")
        input("\nНажмите Enter для продолжения...")
        return
    
    # Создание предиктора
    predictor = GastritisPredictor()
    predictor.load()
    
    # Интерактивная диагностика
    result = predictor.interactive_predict()
    
    # Визуализация вероятности
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_prediction_probability(
        result['probability'],
        threshold=result['threshold'],
        save_path=f'{output_dir}/prediction_{timestamp}.png'
    )
    
    print(f"\nГрафик вероятности сохранен в {output_dir}/prediction_{timestamp}.png")
    
    input("\nНажмите Enter для продолжения...")


def compare_patients_menu():
    """Меню сравнения симптомов двух пациентов."""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ СИМПТОМОВ ДВУХ ПАЦИЕНТОВ")
    print("=" * 70)
    
    # Проверка наличия модели
    if not os.path.exists('models/gastritis_model.keras'):
        print("\nОШИБКА: Обученная модель не найдена!")
        print("Пожалуйста, сначала обучите модель (пункт 1 в главном меню).")
        input("\nНажмите Enter для продолжения...")
        return
    
    # Создание предиктора
    predictor = GastritisPredictor()
    predictor.load()
    
    # Ввод симптомов первого пациента
    print("\nПациент 1:")
    symptoms1 = {}
    for i, (symptom_en, symptom_ru) in enumerate(zip(predictor.symptoms, predictor.symptoms_ru), 1):
        while True:
            try:
                value = input(f"  {i}. {symptom_ru} (0-10): ")
                value = float(value)
                if 0 <= value <= 10:
                    symptoms1[symptom_en] = value
                    break
                else:
                    print("    Значение должно быть от 0 до 10.")
            except ValueError:
                print("    Пожалуйста, введите число от 0 до 10.")
    
    # Ввод симптомов второго пациента
    print("\nПациент 2:")
    symptoms2 = {}
    for i, (symptom_en, symptom_ru) in enumerate(zip(predictor.symptoms, predictor.symptoms_ru), 1):
        while True:
            try:
                value = input(f"  {i}. {symptom_ru} (0-10): ")
                value = float(value)
                if 0 <= value <= 10:
                    symptoms2[symptom_en] = value
                    break
                else:
                    print("    Значение должно быть от 0 до 10.")
            except ValueError:
                print("    Пожалуйста, введите число от 0 до 10.")
    
    # Предсказание для обоих пациентов
    result1 = predictor.predict_from_symptoms(symptoms1)
    result2 = predictor.predict_from_symptoms(symptoms2)
    
    # Вывод результатов
    print("\n" + "-" * 70)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("-" * 70)
    
    print(f"\nПациент 1:")
    print(f"  Диагноз: {result1['prediction']}")
    print(f"  Вероятность: {result1['probability']:.2%}")
    print(f"  Уверенность: {result1['confidence']}")
    
    print(f"\nПациент 2:")
    print(f"  Диагноз: {result2['prediction']}")
    print(f"  Вероятность: {result2['probability']:.2%}")
    print(f"  Уверенность: {result2['confidence']}")
    
    # Визуализация сравнения
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_symptoms_comparison(
        symptoms1, symptoms2,
        'Пациент 1', 'Пациент 2',
        save_path=f'{output_dir}/comparison_{timestamp}.png'
    )
    
    print(f"\nГрафик сравнения сохранен в {output_dir}/comparison_{timestamp}.png")
    
    input("\nНажмите Enter для продолжения...")


def model_info_menu():
    """Меню информации о модели."""
    print("\n" + "=" * 70)
    print("ИНФОРМАЦИЯ О МОДЕЛИ")
    print("=" * 70)
    
    # Проверка наличия модели
    if not os.path.exists('models/gastritis_model.keras'):
        print("\nОбученная модель не найдена.")
        print("Пожалуйста, сначала обучите модель (пункт 1 в главном меню).")
    else:
        print("\nМодель найдена: models/gastritis_model.keras")
        
        # Загрузка модели
        model = GastritisModel()
        model.load_model()
        
        print("\nАрхитектура модели:")
        print(model.get_model_summary())
        
        # Проверка скалера
        if os.path.exists('models/scaler.pkl'):
            print("\nСкалер найден: models/scaler.pkl")
        else:
            print("\nСкалер не найден: models/scaler.pkl")
        
        # Проверка данных
        if os.path.exists('data/gastritis_data.csv'):
            import pandas as pd
            df = pd.read_csv('data/gastritis_data.csv')
            print(f"\nДанные найдены: data/gastritis_data.csv")
            print(f"  Количество образцов: {len(df)}")
            print(f"  Количество признаков: {len(df.columns) - 1}")
            print(f"  Распределение диагнозов:")
            print(df['gastritis'].value_counts().to_string())
        else:
            print("\nДанные не найдены: data/gastritis_data.csv")
        
        # Проверка визуализаций
        if os.path.exists('output'):
            files = os.listdir('output')
            if files:
                print(f"\nВизуализации найдены в директории output/:")
                for file in files:
                    print(f"  - {file}")
            else:
                print("\nВизуализации не найдены.")
        else:
            print("\nДиректория output/ не найдена.")
    
    input("\nНажмите Enter для продолжения...")


def show_examples_menu():
    """Меню показа примеров визуализации."""
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ВИЗУАЛИЗАЦИИ")
    print("=" * 70)
    
    print("\nСоздание примеров визуализации...")
    
    # Создание примеров
    import numpy as np
    
    # 1. График истории обучения
    class MockHistory:
        def __init__(self):
            self.history = {
                'loss': [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24],
                'val_loss': [0.85, 0.7, 0.6, 0.55, 0.5, 0.48, 0.47, 0.46, 0.45, 0.45],
                'accuracy': [0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88],
                'val_accuracy': [0.55, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.76, 0.77, 0.77],
                'precision': [0.65, 0.72, 0.78, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.9],
                'val_precision': [0.6, 0.68, 0.73, 0.75, 0.77, 0.78, 0.79, 0.79, 0.8, 0.8],
                'recall': [0.55, 0.68, 0.72, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85, 0.86],
                'val_recall': [0.5, 0.62, 0.67, 0.7, 0.72, 0.73, 0.74, 0.74, 0.75, 0.75]
            }
    
    history = MockHistory()
    plot_training_history(history, save_path='output/example_training_history.png')
    print("  ✓ График истории обучения")
    
    # 2. Матрица ошибок
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
    plot_confusion_matrix(y_true, y_pred, save_path='output/example_confusion_matrix.png')
    print("  ✓ Матрица ошибок")
    
    # 3. Важность признаков
    weights = np.random.randn(10, 64)
    feature_names = [
        'abdominal_pain', 'heartburn', 'nausea', 'vomiting', 'loss_of_appetite',
        'bloating', 'belching', 'weakness', 'weight_loss', 'discomfort_after_meal'
    ]
    plot_feature_importance(weights, feature_names, save_path='output/example_feature_importance.png')
    print("  ✓ График важности признаков")
    
    # 4. Сравнение симптомов
    symptoms1 = {
        'abdominal_pain': 8, 'heartburn': 7, 'nausea': 6, 'vomiting': 4,
        'loss_of_appetite': 7, 'bloating': 6, 'belching': 5, 'weakness': 4,
        'weight_loss': 3, 'discomfort_after_meal': 8
    }
    symptoms2 = {
        'abdominal_pain': 2, 'heartburn': 1, 'nausea': 1, 'vomiting': 0,
        'loss_of_appetite': 1, 'bloating': 2, 'belching': 1, 'weakness': 1,
        'weight_loss': 0, 'discomfort_after_meal': 2
    }
    plot_symptoms_comparison(symptoms1, symptoms2, 'Пациент с гастритом', 'Здоровый пациент',
                           save_path='output/example_symptoms_comparison.png')
    print("  ✓ График сравнения симптомов")
    
    # 5. Вероятность предсказания
    plot_prediction_probability(0.85, threshold=0.5, 
                              save_path='output/example_prediction_probability.png')
    print("  ✓ График вероятности предсказания")
    
    # 6. Метрики модели
    metrics = {
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.88,
        'f1_score': 0.89
    }
    plot_metrics_comparison(metrics, save_path='output/example_metrics_comparison.png')
    print("  ✓ График метрик модели")
    
    print("\nВсе примеры визуализации сохранены в директорию output/")
    
    input("\nНажмите Enter для продолжения...")


def main():
    """Главная функция приложения."""
    print_header()
    
    while True:
        print_menu()
        choice = input("\nВаш выбор (1-6): ")
        
        if choice == '1':
            train_model_menu()
        elif choice == '2':
            diagnose_patient_menu()
        elif choice == '3':
            compare_patients_menu()
        elif choice == '4':
            model_info_menu()
        elif choice == '5':
            show_examples_menu()
        elif choice == '6':
            print("\nСпасибо за использование системы диагностики гастрита!")
            print("Помните: для точного диагноза обязательно проконсультируйтесь с врачом.")
            print("\nДо свидания!\n")
            break
        else:
            print("\nНеверный выбор. Пожалуйста, введите число от 1 до 6.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        sys.exit(1)
