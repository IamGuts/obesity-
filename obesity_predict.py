import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class ObesityClassifier:
    def __init__(self, model_path):
        """
        Инициализация классификатора.
        :param model_path: Путь к файлу модели.
        """
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Загрузка модели из файла."""
        try:
            self.model = joblib.load(self.model_path)
            
            # Проверка, что загруженный объект является Pipeline или моделью
            if not isinstance(self.model, (Pipeline, BaseEstimator)):
                raise TypeError(f"Загруженный объект не является Pipeline или моделью scikit-learn. Тип объекта: {type(self.model)}")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл модели не найден по пути: {self.model_path}")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке модели: {e}")

    def predict(self, input_data):
        """
        Предсказание на основе входных данных.
        :param input_data: Данные для предсказания в формате DataFrame или массива.
        :return: Результат предсказания.
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Убедитесь, что модель была успешно загружена.")
        
        try:
            return self.model.predict(input_data)
        except AttributeError:
            raise AttributeError("Загруженный объект не имеет метода 'predict'. Убедитесь, что это модель машинного обучения.")
        except Exception as e:
            raise Exception(f"Ошибка при выполнении предсказания: {e}")