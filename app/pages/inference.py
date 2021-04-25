import streamlit as st
from sqlalchemy import case, desc
from sqlalchemy.orm import Query

from app import db
from app.db.utils import create_session, get_dataframe_from_query
from app.model.inference import main


def app():
    st.markdown(
        """
    *Получение предсказаний модели*
    """
    )
    df = get_dataframe_from_query(
        Query(
            [
                db.Model.name.label('Модель'),
                db.Model.description.label('Описание'),
                db.Model.params.label('Параметры модели'),
                db.Model.metrics.label('Метрики'),
                db.Dataset.name.label('Датасет'),
                case(
                    [(db.Model.pretrained, 'Да'), (~db.Model.pretrained, 'Нет')]
                ).label('Предобученная'),
                db.Model.training_time.label('Время обучения, с'),
                db.Model.created_at.label('Дата обучения'),
            ]
        )
        .join(db.Project)
        .join(db.Dataset)
        .order_by(desc(db.Model.created_at))
    )

    st.dataframe(df)

    selected_model = st.selectbox('Выбор модели', df['Модель'].unique())
    photo = st.file_uploader('Фотография', type=['png', 'jpg', 'jpeg'])
    if photo:
        st.image(photo)

    predict_button = st.button('Предсказать')

    if predict_button:
        if not photo:
            st.error('Необходимо загрузить фото')
        else:
            with create_session() as session:
                model_path = (
                    session.query(db.Model.path)
                    .filter(db.Model.name == selected_model)
                    .first()[0]
                )

        with open('temp.png', 'wb') as f:
            f.write(photo.read())

        try:
            kwargs = {
                'path': 'temp.png',
                'n_crops': 228,
                'path_out': 'res.png',
                'top_k': 3,
                'type': 'window',  # or 'random'
                'canny_crop': False,
                'model_path': str(model_path)
            }
            main(**kwargs)
            st.image('res.png')

        except Exception as e:
            st.exception(e)
