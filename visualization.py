"""
Модуль для визуализации результатов обучения и предсказаний модели.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(history, save_path=None):
    """
    Построение графиков истории обучения.
    
    Args:
        history: Объект истории обучения Keras
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('История обучения модели', fontsize=16, fontweight='bold')
    
    # График потерь
    axes[0, 0].plot(history.history['loss'], label='Обучающая выборка', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Валидационная выборка', linewidth=2)
    axes[0, 0].set_title('Потери (Loss)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Эпоха', fontsize=10)
    axes[0, 0].set_ylabel('Потеря', fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # График точности
    axes[0, 1].plot(history.history['accuracy'], label='Обучающая выборка', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Валидационная выборка', linewidth=2)
    axes[0, 1].set_title('Точность (Accuracy)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Эпоха', fontsize=10)
    axes[0, 1].set_ylabel('Точность', fontsize=10)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # График Precision
    axes[1, 0].plot(history.history['precision'], label='Обучающая выборка', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Валидационная выборка', linewidth=2)
    axes[1, 0].set_title('Точность (Precision)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Эпоха', fontsize=10)
    axes[1, 0].set_ylabel('Precision', fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # График Recall
    axes[1, 1].plot(history.history['recall'], label='Обучающая выборка', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Валидационная выборка', linewidth=2)
    axes[1, 1].set_title('Полнота (Recall)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Эпоха', fontsize=10)
    axes[1, 1].set_ylabel('Recall', fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График истории обучения сохранен в {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Построение матрицы ошибок.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        save_path: Путь для сохранения графика
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Нет гастрита', 'Гастрит'],
                yticklabels=['Нет гастрита', 'Гастрит'],
                cbar_kws={'label': 'Количество образцов'})
    
    plt.title('Матрица ошибок (Confusion Matrix)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Предсказанный диагноз', fontsize=12)
    plt.ylabel('Истинный диагноз', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Матрица ошибок сохранена в {save_path}")
    
    plt.close()


def plot_feature_importance(weights, feature_names, save_path=None):
    """
    Построение графика важности признаков.
    
    Args:
        weights: Веса первого слоя модели
        feature_names: Названия признаков
        save_path: Путь для сохранения графика
    """
    # Вычисляем важность признаков как сумму абсолютных значений весов
    importance = np.abs(weights).sum(axis=1)
    
    # Сортируем по важности
    indices = np.argsort(importance)[::-1]
    
    # Русские названия симптомов
    feature_names_ru = [
        'Боль в животе',
        'Изжога',
        'Тошнота',
        'Рвота',
        'Потеря аппетита',
        'Вздутие живота',
        'Отрыжка',
        'Слабость',
        'Потеря веса',
        'Дискомфорт после еды'
    ]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(importance)), importance[indices], color='steelblue')
    
    # Добавляем значения на бары
    for i, (idx, bar) in enumerate(zip(indices, bars)):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{importance[idx]:.2f}', 
                ha='left', va='center', fontsize=9)
    
    plt.yticks(range(len(importance)), [feature_names_ru[i] for i in indices])
    plt.xlabel('Важность признака', fontsize=12)
    plt.title('Важность симптомов для диагностики гастрита', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График важности признаков сохранен в {save_path}")
    
    plt.close()


def plot_symptoms_comparison(symptoms_dict1, symptoms_dict2, 
                             label1='Пациент 1', label2='Пациент 2', 
                             save_path=None):
    """
    Сравнительный график симптомов двух пациентов.
    
    Args:
        symptoms_dict1: Словарь симптомов первого пациента
        symptoms_dict2: Словарь симптомов второго пациента
        label1: Метка первого пациента
        label2: Метка второго пациента
        save_path: Путь для сохранения графика
    """
    symptoms_ru = [
        'Боль в животе',
        'Изжога',
        'Тошнота',
        'Рвота',
        'Потеря аппетита',
        'Вздутие живота',
        'Отрыжка',
        'Слабость',
        'Потеря веса',
        'Дискомфорт после еды'
    ]
    
    symptoms_en = [
        'abdominal_pain',
        'heartburn',
        'nausea',
        'vomiting',
        'loss_of_appetite',
        'bloating',
        'belching',
        'weakness',
        'weight_loss',
        'discomfort_after_meal'
    ]
    
    values1 = [symptoms_dict1.get(symptom, 0) for symptom in symptoms_en]
    values2 = [symptoms_dict2.get(symptom, 0) for symptom in symptoms_en]
    
    x = np.arange(len(symptoms_ru))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, values1, width, label=label1, color='steelblue')
    bars2 = ax.bar(x + width/2, values2, width, label=label2, color='coral')
    
    ax.set_xlabel('Симптомы', fontsize=12)
    ax.set_ylabel('Интенсивность (0-10)', fontsize=12)
    ax.set_title('Сравнение симптомов пациентов', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(symptoms_ru, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на бары
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения симптомов сохранен в {save_path}")
    
    plt.close()


def plot_prediction_probability(probability, threshold=0.5, save_path=None):
    """
    Визуализация вероятности предсказания.
    
    Args:
        probability: Вероятность наличия гастрита
        threshold: Порог классификации
        save_path: Путь для сохранения графика
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Создаем градиентную шкалу
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 1, 0, 1])
    
    # Добавляем маркер текущей вероятности
    ax.axvline(x=probability, color='black', linewidth=3, linestyle='--')
    ax.axvline(x=threshold, color='red', linewidth=2, linestyle=':', label='Порог классификации')
    
    # Добавляем текст
    ax.text(probability, 0.5, f'{probability:.1%}', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Вероятность гастрита', fontsize=12)
    ax.set_title('Вероятность диагноза', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    
    # Добавляем подписи зон
    ax.text(threshold/2, 0.1, 'НЕТ ГАСТРИТА', ha='center', fontsize=10, fontweight='bold')
    ax.text((threshold + 1)/2, 0.1, 'ГАСТРИТ', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График вероятности сохранен в {save_path}")
    
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    График сравнения метрик модели.
    
    Args:
        metrics_dict: Словарь с метриками
        save_path: Путь для сохранения графика
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics_dict.get('accuracy', 0),
        metrics_dict.get('precision', 0),
        metrics_dict.get('recall', 0),
        metrics_dict.get('f1_score', 0)
    ]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Добавляем значения на бары
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Значение метрики', fontsize=12)
    ax.set_title('Метрики качества модели', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График метрик сохранен в {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Пример использования функций визуализации
    
    print("Примеры визуализации:")
    
    # 1. График истории обучения (имитация)
    print("\n1. Создание примера графика истории обучения...")
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
    
    # 2. Матрица ошибок
    print("2. Создание примера матрицы ошибок...")
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
    plot_confusion_matrix(y_true, y_pred, save_path='output/example_confusion_matrix.png')
    
    # 3. Важность признаков
    print("3. Создание примера графика важности признаков...")
    weights = np.random.randn(10, 64)
    feature_names = [
        'abdominal_pain', 'heartburn', 'nausea', 'vomiting', 'loss_of_appetite',
        'bloating', 'belching', 'weakness', 'weight_loss', 'discomfort_after_meal'
    ]
    plot_feature_importance(weights, feature_names, save_path='output/example_feature_importance.png')
    
    # 4. Сравнение симптомов
    print("4. Создание примера графика сравнения симптомов...")
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
    
    # 5. Вероятность предсказания
    print("5. Создание примера графика вероятности...")
    plot_prediction_probability(0.85, threshold=0.5, 
                              save_path='output/example_prediction_probability.png')
    
    # 6. Метрики модели
    print("6. Создание примера графика метрик...")
    metrics = {
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.88,
        'f1_score': 0.89
    }
    plot_metrics_comparison(metrics, save_path='output/example_metrics_comparison.png')
    
    print("\nВсе примеры визуализации сохранены в директорию output/")
