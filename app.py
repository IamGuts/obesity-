import streamlit as st
import pandas as pd 

from Obesity_py import ObesityClassifier
from config import MODEL_PATH, IMAGE_PATH


with st.sidebar:
    st.markdown('<div style="background-color: red; padding: 10px; color: white;">Параметры ввода</div>', unsafe_allow_html=True)
    st.title('Классификация ожирения')
    st.write('Введете данные для пресказания уровня ожирения')


    # Поля для ввода даных
    age = st.number_input('Возраст', min_value=1.0, max_value=120.0, value=25.0, step=1.0)
    weight = st.number_input('Вес (кг)',min_value=10.0, max_value=300.0, value = 70.0, step = 1.0)
    height = st.number_input('Рот (см)', min_value=50.0, max_value= 250.0, value=170.0, step = 1.0)

    # Выбор пола 
    gender = st.radio('Пол', options= ['Male', 'Female'])

    # Кол-во основных приемов пищи (ползунок)
    ncp = st.slider('Колличетво основных приемов пищи', min_value=1.0, max_value=4.0, value=3.0, step=0.5)

    # Курение ( галочка )
    smoke = st.checkbox('курение', value=False)

    # Потребление воды (ползунок)
    ch2o = st.slider("Потребление воды (литры в день)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    # Физ Активность (ползунок)

    faf = st.slider('Физическая активность (часы в день)', min_value=0.0, max_value=3.0, value=1.0, step=0.1)

    # Употребление алкоголя ( ползунок )
    calc = st.slider('Употребление алкоголя (раз в неделю)', min_value=0.0, max_value=4.0, value=1.0, step= 0.5)

    # Тип транспорта (выбор из списка)
    mtrans = st.selectbox(
        'Тип Транспорта',
        options=["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"]
    )

    # Конопка для предсказаний

    if st.button('Предсказать'):
        # инициация классификатора
        classifier = ObesityClassifier(model_path=MODEL_PATH)

        # Подготовка данных 
        input_data = {
            'Age': [age],  # Обратите внимание: значения передаются как списки
            'Weight': [weight],
            'Height' : [height],
            'IMT' : [weight / height ** 2],
            'Gender': [gender],
            'family_history': ['yes'],  # Пример: есть семейная история
            'FAVC': ['yes'],  # Частое употребление высококалорийной пищи
            'FCVC': [2.0],  # Частота употребления овощей
            'NCP': [ncp],  # Количество основных приемов пищи
            'CAEC': ['Sometimes'],  # Употребление пищи между приемами
            'SMOKE': ['yes' if smoke else 'no'],  # Курение
            'CH2O': [ch2o],  # Потребление воды
            'SCC': ['no'],  # Мониторинг калорий
            'FAF': [faf],  # Физическая активность
            'TUE': [0.0],  # Время использования электронных устройств
            'CALC': ['Sometimes' if calc > 0 else 'no'],  # Употребление алкоголя
            'MTRANS': [mtrans]  # Тип транспорта
        }



        # Преобразуем данные в DataFrame
        input_df = pd.DataFrame(input_data)

        # Выполнение предсказания
        result = classifier.model.predict(input_df)
        result_list = [
        'Нормальный вес',
        'Избыточный вес I уровня', 
        'Избыточный вес II уровня',
        'Ожирение I типа',
        'Недостаточный вес',
        'Ожирение II типа',
        'Ожирение III типа'
        ]
        # jnj,hf;tybt htpekmnfnf
        if result is not None:
            st.success(f'Резульатат предсказания : {result_list[result[0]]}')
        else:
            st.error('Не удалось выполнить предсказние.')

st.image(IMAGE_PATH,caption='Описание изображения', use_container_width=True)