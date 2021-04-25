import streamlit as st
from sqlalchemy.orm import Query

from app import db
from app.db.utils import create_session, get_dataframe_from_query
from app.model.train_entry import main


def app():
    df = get_dataframe_from_query(Query(db.Dataset))
    st.dataframe(df)
    st.markdown(
        """
    ```
    *Пример конфига*
    {
        'device': 5,
        'dataset_path': 'data/',
        'batch_size': 4,
        'pretrained': True,
        'num_classes': 3,
        'checkpoint_path': 'train_entry_check',
        'logdir': 'train_entry_log'
    }
    ```
    """
    )
    name = st.text_input('Название модели')
    description = st.text_input('Описание модели')
    st.text_area('Конфиг модели')

    selected_dataset = st.selectbox('Выбор датасета', df['name'].unique())
    if selected_dataset:
        with create_session() as session:
            dataset_id = (
                session.query(db.Dataset.id)
                .filter(db.Dataset.name == selected_dataset)
                .first()[0]
            )

    train_button = st.button('Обучить')
    checkpoint_path = 'train_entry_check'
    print(dataset_id)
    if train_button:
        kwargs = {
            'device': 0,
            'dataset_path': f'data/{str(dataset_id)}',
            'batch_size': 1,
            'pretrained': True,
            'num_classes': 3,
            'checkpoint_path': checkpoint_path,
            'logdir': 'train_entry_log',
        }
        checkpoint_path = main(**kwargs)
        st.info('Модель обучилась')

        with create_session() as session:
            session.add(
                db.Model(
                    name=name,
                    description=description,
                    path=checkpoint_path,
                    dataset_id=dataset_id,
                    project_id=1,
                )
            )
