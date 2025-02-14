import streamlit as st
import pandas as pd
from obesity_predict import ObesityClassifier
from config import MODEL_PATH, IMAGE_PATH
import json
import os 
from pathlib import Path 


# Загрузка описаний
DESC_PATH = Path(__file__).resolve().parent / "text" / "descriptions.json"

try:
    with open(DESC_PATH, "r", encoding="utf-8") as f:
        class_descriptions = json.load(f)
except Exception as e:
    st.error(f"Ошибка загрузки описаний: {str(e)}")


def main():
    st.title('Классификация ожирения')
    st.write('Введите данные для предсказания уровня ожирения')

    with st.sidebar:
        st.markdown('<div style="background-color: red; padding: 10px; color: white;">Параметры ввода</div>', unsafe_allow_html=True)
        st.title('Классификация ожирения')
        st.write('Введите данные для предсказания уровня ожирения')

        age = st.number_input('Возраст', min_value=1.0, max_value=120.0, value=25.0, step=1.0)
        weight = st.number_input('Вес (кг)', min_value=10.0, max_value=300.0, value=70.0, step=1.0)
        height = st.number_input('Рост (см)', min_value=50.0, max_value=250.0, value=170.0, step=1.0)
        gender = st.radio('Пол', options=['Male', 'Female'])
        ncp = st.slider('Количество основных приемов пищи', min_value=1.0, max_value=4.0, value=3.0, step=0.5)
        smoke = st.checkbox('Курение', value=False)
        ch2o = st.slider("Потребление воды (литры в день)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        faf = st.slider('Физическая активность (часы в день)', min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        calc = st.slider('Употребление алкоголя (раз в неделю)', min_value=0.0, max_value=4.0, value=1.0, step=0.5)
        mtrans = st.selectbox(
            'Тип Транспорта',
            options=["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"]
        )

    if st.button('Предсказать'):
        try:
            classifier = ObesityClassifier(model_path=MODEL_PATH)

            input_data = {
                'Age': [age],
                'Weight': [weight],
                'Height': [height],
                'IMT': [weight / (height / 100) ** 2],
                'Gender': [gender],
                'family_history': ['yes'],
                'FAVC': ['yes'],
                'FCVC': [2.0],
                'NCP': [ncp],
                'CAEC': ['Sometimes'],
                'SMOKE': ['yes' if smoke else 'no'],
                'CH2O': [ch2o],
                'SCC': ['no'],
                'FAF': [faf],
                'TUE': [0.0],
                'CALC': ['Sometimes' if calc > 0 else 'no'],
                'MTRANS': [mtrans]
            }

            input_df = pd.DataFrame(input_data)
            result = classifier.predict(input_df)
            encod_params =  {
                'Normal_Weight' : 0,
                'Overweight_Level_I' : 1,
                'Overweight_Level_II' : 2,
                'Obesity_Type_I' : 3,
                'Insufficient_Weight' : 4,
                'Obesity_Type_II' : 5,
                'Obesity_Type_III' : 6
            }
            encod_params_inv = {v: k for k, v in encod_params.items()}

            if result is not None:
                class_name = encod_params_inv[result[0]]
                description = class_descriptions.get(class_name, "Описание не найдено")
                st.success(f"**Результат:** {description}")

                st.success(f'Результат предсказания: {result_text.[
                    encod_params_inv[
                        result[0]
                        ]
                    ]
                    }')
            else:
                st.error('Не удалось выполнить предсказание.')
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")

    # Отображение изображения
    try:
        st.image(IMAGE_PATH, caption='Описание изображения', use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения: {e}")

if __name__ == "__main__":
    main()