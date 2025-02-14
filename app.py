import streamlit as st
import pandas as pd
from obesity_predict import ObesityClassifier
from config import MODEL_PATH, IMAGE_PATH
import json
import os 



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

            describe_result ={
    'Normal_Weight' : 'Здоровый вес: ИМТ в диапазоне 18.5-24.9, оптимальное соотношение мышечной массы и жира.',
    'Overweight_Level_I' : 'Предожирение: ИМТ 25-29.9, повышенная нагрузка на суставы, риск развития метаболических нарушений.',
    'Overweight_Level_II' : 'Умеренный избыток веса: ИМТ 30-34.9 (часто классифицируется как ожирение I степени), начало жировых отложений в абдоминальной зоне.',
    'Obesity_Type_I' : 'Ожирение I степени: ИМТ 35-39.9, повышенный риск диабета 2 типа и гипертонии.',
    'Insufficient_Weight' : 'Дефицит массы: ИМТ <18.5, возможны проблемы с метаболизмом и иммунитетом.',
    'Obesity_Type_II' : 'Ожирение II степени: ИМТ 40-44.9, выраженные жировые отложения, высокая нагрузка на сердечно-сосудистую систему.',
    'Obesity_Type_III' : ' Морбидное ожирение: ИМТ ≥45, критический риск сопутствующих патологий (апноэ, артроз).'
}

            if result is not None:
                result_text = os.path.join(os.path.dirname(__file__), "text", "describe.py")

                st.success(encod_params_inv[result[0]])
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