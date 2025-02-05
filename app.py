import streamlit as st
import pandas as pd
from obesity_predict import ObesityClassifier
from config import MODEL_PATH, IMAGE_PATH

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
            result_list = [
                'Нормальный вес',
                'Избыточный вес I уровня',
                'Избыточный вес II уровня',
                'Ожирение I типа',
                'Недостаточный вес',
                'Ожирение II типа',
                'Ожирение III типа'
            ]

            if result is not None:
                st.success(f'Результат предсказания: {result_list[result[0]]}')
            else:
                st.error('Не удалось выполнить предсказание.')
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")

    # Отображение изображения
    #st.image(IMAGE_PATH, caption='Описание изображения', use_container_width=True)

if __name__ == "__main__":
    main()