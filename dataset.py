"""
Модуль для генерации и подготовки данных симптомов для диагностики гастрита.
Симптомы включают: боль в животе, изжога, тошнота, рвота, потеря аппетита,
вздутие живота, отрыжка, слабость, потеря веса, дискомфорт после еды.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


class GastritisDataset:
    """Класс для работы с датасетом симптомов гастрита."""
    
    def __init__(self, random_state=42):
        """
        Инициализация датасета.
        
        Args:
            random_state: Сид для воспроизводимости результатов
        """
        self.random_state = random_state
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
        self.target = 'gastritis'  # 0 - нет гастрита, 1 - есть гастрит
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Генерация синтетических данных симптомов.
        
        Args:
            n_samples: Количество образцов для генерации
            
        Returns:
            DataFrame с симптомами и диагнозом
        """
        np.random.seed(self.random_state)
        
        data = []
        
        for _ in range(n_samples):
            # Случайно выбираем, есть ли у пациента гастрит
            has_gastritis = np.random.choice([0, 1], p=[0.4, 0.6])
            
            if has_gastritis:
                # Пациенты с гастритом имеют более выраженные симптомы
                abdominal_pain = np.random.normal(7, 2)
                heartburn = np.random.normal(6, 2)
                nausea = np.random.normal(5, 2)
                vomiting = np.random.normal(3, 2)
                loss_of_appetite = np.random.normal(6, 2)
                bloating = np.random.normal(5, 2)
                belching = np.random.normal(5, 2)
                weakness = np.random.normal(4, 2)
                weight_loss = np.random.normal(3, 2)
                discomfort_after_meal = np.random.normal(7, 2)
            else:
                # Пациенты без гастрита имеют менее выраженные симптомы
                abdominal_pain = np.random.normal(2, 1.5)
                heartburn = np.random.normal(2, 1.5)
                nausea = np.random.normal(1, 1)
                vomiting = np.random.normal(0.5, 0.5)
                loss_of_appetite = np.random.normal(1, 1)
                bloating = np.random.normal(2, 1.5)
                belching = np.random.normal(2, 1.5)
                weakness = np.random.normal(1, 1)
                weight_loss = np.random.normal(0.5, 0.5)
                discomfort_after_meal = np.random.normal(2, 1.5)
            
            # Ограничиваем значения от 0 до 10
            symptoms = [
                max(0, min(10, abdominal_pain)),
                max(0, min(10, heartburn)),
                max(0, min(10, nausea)),
                max(0, min(10, vomiting)),
                max(0, min(10, loss_of_appetite)),
                max(0, min(10, bloating)),
                max(0, min(10, belching)),
                max(0, min(10, weakness)),
                max(0, min(10, weight_loss)),
                max(0, min(10, discomfort_after_meal))
            ]
            
            data.append(symptoms + [has_gastritis])
        
        df = pd.DataFrame(data, columns=self.symptoms + [self.target])
        return df
    
    def prepare_data(self, df=None, test_size=0.2, val_size=0.1):
        """
        Подготовка данных для обучения модели.
        
        Args:
            df: DataFrame с данными (если None, генерирует новые)
            test_size: Размер тестовой выборки
            val_size: Размер валидационной выборки
            
        Returns:
            Словарь с подготовленными данными и скалером
        """
        if df is None:
            df = self.generate_synthetic_data()
        
        # Разделяем признаки и целевую переменную
        X = df[self.symptoms].values
        y = df[self.target].values
        
        # Нормализация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Разделение на обучающую, валидационную и тестовую выборки
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=(test_size + val_size), 
            random_state=self.random_state, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + val_size),
            random_state=self.random_state, stratify=y_temp
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': self.symptoms
        }
    
    def save_data(self, df, filepath='data/gastritis_data.csv'):
        """
        Сохранение данных в CSV файл.
        
        Args:
            df: DataFrame для сохранения
            filepath: Путь к файлу
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Данные сохранены в {filepath}")
    
    def load_data(self, filepath='data/gastritis_data.csv'):
        """
        Загрузка данных из CSV файла.
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            DataFrame с данными
        """
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"Данные загружены из {filepath}")
            return df
        else:
            print(f"Файл {filepath} не найден. Генерирую новые данные.")
            return self.generate_synthetic_data()


if __name__ == "__main__":
    # Пример использования
    dataset = GastritisDataset()
    
    # Генерация данных
    df = dataset.generate_synthetic_data(n_samples=1000)
    print(f"Сгенерировано {len(df)} образцов")
    print(f"\nПервые 5 строк:")
    print(df.head())
    print(f"\nСтатистика по диагнозу:")
    print(df['gastritis'].value_counts())
    
    # Сохранение данных
    dataset.save_data(df)
    
    # Подготовка данных для обучения
    prepared_data = dataset.prepare_data(df)
    print(f"\nРазмеры выборок:")
    print(f"Обучающая: {prepared_data['X_train'].shape}")
    print(f"Валидационная: {prepared_data['X_val'].shape}")
    print(f"Тестовая: {prepared_data['X_test'].shape}")
