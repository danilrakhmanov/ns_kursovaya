"""
Скрипт для предсказания диагноза гастрита на основе симптомов пациента.
"""

import numpy as np
from model import GastritisModel
from dataset import GastritisDataset


class GastritisPredictor:
    """Класс для предсказания диагноза гастрита."""
    
    def __init__(self, model_path='models/gastritis_model.keras', 
                 scaler_path='models/scaler.pkl'):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к файлу модели
            scaler_path: Путь к файлу скалера
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.symptoms = [
            'abdominal_pain',      # Боль в животе (0-10)
            'heartburn',           # Изжога (0-10)
            'nausea',              # Тошнота (0-10)
            'vomiting',            # Рвота (0-10)
            'loss_of_appetite',    # Потеря аппетита (0-10)
            'bloating',            # Вздутие живота (0-10)
            'belching',            # Отрыжка (0-10)
            'weakness',            # Слабость (0-10)
            'weight_loss',         # Потеря веса (0-10)
            'discomfort_after_meal' # Дискомфорт после еды (0-10)
        ]
        self.symptoms_ru = [
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
        
    def load(self):
        """Загрузка модели и скалера."""
        print("Загрузка модели и скалера...")
        self.model = GastritisModel()
        self.model.load_model(self.model_path)
        self.scaler = self.model.load_scaler(self.scaler_path)
        print("Модель и скалер загружены успешно!")
    
    def predict_from_symptoms(self, symptoms_dict, threshold=0.5):
        """
        Предсказание диагноза на основе словаря симптомов.
        
        Args:
            symptoms_dict: Словарь с симптомами (ключи - названия симптомов, значения - 0-10)
            threshold: Порог для классификации
            
        Returns:
            Словарь с результатами предсказания
        """
        if self.model is None or self.scaler is None:
            self.load()
        
        # Формирование вектора симптомов
        symptoms_vector = []
        for symptom in self.symptoms:
            value = symptoms_dict.get(symptom, 0)
            symptoms_vector.append(value)
        
        # Нормализация
        symptoms_array = np.array([symptoms_vector])
        symptoms_scaled = self.scaler.transform(symptoms_array)
        
        # Предсказание
        prediction, probability = self.model.predict(symptoms_scaled, threshold)
        
        # Формирование результата
        result = {
            'prediction': 'Гастрит' if prediction == 1 else 'Нет гастрита',
            'probability': float(probability),
            'confidence': 'Высокая' if probability > 0.8 else 'Средняя' if probability > 0.6 else 'Низкая',
            'symptoms': symptoms_dict,
            'threshold': threshold
        }
        
        return result
    
    def predict_from_array(self, symptoms_array, threshold=0.5):
        """
        Предсказание диагноза на основе массива симптомов.
        
        Args:
            symptoms_array: Массив с симптомами (10 значений от 0 до 10)
            threshold: Порог для классификации
            
        Returns:
            Словарь с результатами предсказания
        """
        if self.model is None or self.scaler is None:
            self.load()
        
        # Нормализация
        symptoms_scaled = self.scaler.transform([symptoms_array])
        
        # Предсказание
        prediction, probability = self.model.predict(symptoms_scaled, threshold)
        
        # Формирование словаря симптомов
        symptoms_dict = dict(zip(self.symptoms, symptoms_array))
        
        # Формирование результата
        result = {
            'prediction': 'Гастрит' if prediction == 1 else 'Нет гастрита',
            'probability': float(probability),
            'confidence': 'Высокая' if probability > 0.8 else 'Средняя' if probability > 0.6 else 'Низкая',
            'symptoms': symptoms_dict,
            'threshold': threshold
        }
        
        return result
    
    def interactive_predict(self):
        """
        Интерактивный режим предсказания с вводом симптомов пользователем.
        """
        if self.model is None or self.scaler is None:
            self.load()
        
        print("\n" + "=" * 60)
        print("ИНТЕРАКТИВНАЯ ДИАГНОСТИКА ГАСТРИТА")
        print("=" * 60)
        print("\nПожалуйста, оцените интенсивность каждого симптома от 0 до 10:")
        print("  0 - симптом отсутствует")
        print("  10 - максимально выраженный симптом")
        print()
        
        symptoms_dict = {}
        for i, (symptom_en, symptom_ru) in enumerate(zip(self.symptoms, self.symptoms_ru), 1):
            while True:
                try:
                    value = input(f"{i}. {symptom_ru} (0-10): ")
                    value = float(value)
                    if 0 <= value <= 10:
                        symptoms_dict[symptom_en] = value
                        break
                    else:
                        print("  Значение должно быть от 0 до 10. Попробуйте снова.")
                except ValueError:
                    print("  Пожалуйста, введите число от 0 до 10.")
        
        # Предсказание
        result = self.predict_from_symptoms(symptoms_dict)
        
        # Вывод результата
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТ ДИАГНОСТИКИ")
        print("=" * 60)
        print(f"\nПредсказание: {result['prediction']}")
        print(f"Вероятность: {result['probability']:.2%}")
        print(f"Уверенность: {result['confidence']}")
        
        print("\nВведенные симптомы:")
        for symptom_en, symptom_ru in zip(self.symptoms, self.symptoms_ru):
            value = result['symptoms'][symptom_en]
            bar = "█" * int(value)
            print(f"  {symptom_ru}: {value}/10 {bar}")
        
        print("\n" + "=" * 60)
        
        # Рекомендации
        if result['prediction'] == 'Гастрит':
            print("\nРЕКОМЕНДАЦИИ:")
            print("  • Обратитесь к врачу-гастроэнтерологу для консультации")
            print("  • Соблюдайте диету (исключите острую, жирную, кислую пищу)")
            print("  • Избегайте алкоголя и курения")
            print("  • Питайтесь дробно, небольшими порциями")
            print("  • Избегайте стрессовых ситуаций")
        else:
            print("\nРЕКОМЕНДАЦИИ:")
            print("  • Продолжайте вести здоровый образ жизни")
            print("  • Соблюдайте режим питания")
            print("  • При появлении симптомов обратитесь к врачу")
        
        print("=" * 60 + "\n")
        
        return result
    
    def batch_predict(self, symptoms_list, threshold=0.5):
        """
        Пакетное предсказание для нескольких пациентов.
        
        Args:
            symptoms_list: Список массивов симптомов
            threshold: Порог для классификации
            
        Returns:
            Список результатов предсказания
        """
        if self.model is None or self.scaler is None:
            self.load()
        
        # Нормализация
        symptoms_scaled = self.scaler.transform(symptoms_list)
        
        # Предсказание
        predictions, probabilities = self.model.predict_batch(symptoms_scaled, threshold)
        
        # Формирование результатов
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            symptoms_dict = dict(zip(self.symptoms, symptoms_list[i]))
            result = {
                'patient_id': i + 1,
                'prediction': 'Гастрит' if pred == 1 else 'Нет гастрита',
                'probability': float(prob),
                'confidence': 'Высокая' if prob > 0.8 else 'Средняя' if prob > 0.6 else 'Низкая',
                'symptoms': symptoms_dict
            }
            results.append(result)
        
        return results


def main():
    """Главная функция для демонстрации работы предиктора."""
    print("Система диагностики гастрита на основе искусственного интеллекта")
    print("=" * 60)
    
    # Создание предиктора
    predictor = GastritisPredictor()
    
    # Проверка наличия обученной модели
    try:
        predictor.load()
    except FileNotFoundError:
        print("\nОШИБКА: Обученная модель не найдена!")
        print("Пожалуйста, сначала запустите train.py для обучения модели.")
        return
    
    # Интерактивный режим
    while True:
        print("\nВыберите режим работы:")
        print("  1. Интерактивная диагностика")
        print("  2. Пример предсказания")
        print("  3. Выход")
        
        choice = input("\nВаш выбор (1-3): ")
        
        if choice == '1':
            predictor.interactive_predict()
        elif choice == '2':
            # Пример с симптомами гастрита
            print("\nПример 1: Пациент с симптомами гастрита")
            symptoms_gastritis = [8, 7, 6, 4, 7, 6, 5, 4, 3, 8]
            result = predictor.predict_from_array(symptoms_gastritis)
            print(f"Предсказание: {result['prediction']}")
            print(f"Вероятность: {result['probability']:.2%}")
            
            print("\nПример 2: Пациент без выраженных симптомов")
            symptoms_healthy = [2, 1, 1, 0, 1, 2, 1, 1, 0, 2]
            result = predictor.predict_from_array(symptoms_healthy)
            print(f"Предсказание: {result['prediction']}")
            print(f"Вероятность: {result['probability']:.2%}")
        elif choice == '3':
            print("\nДо свидания!")
            break
        else:
            print("\nНеверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
