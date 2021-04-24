import logging
import time

import streamlit as st
from sqlalchemy.orm import Query

from app import db
from app.db.utils import create_session, get_dataframe_from_query
from app.pages.utils import Page, render_horizontal_pages
from app.state import session_state

logger = logging.getLogger(__name__)


def app():
    st.markdown("""TODO (v.karmazin) добавить умный текст""")
    pages = [
        Page('## Просмотр датасета', show_dataset),
        Page('## Создать новый датасет', create_dataset),
        Page('## Редактирование датасета', edit_dataset),
    ]
    render_horizontal_pages(pages, session_state=session_state)


def create_dataset():
    st.markdown(
        '*Необходимо дать название датасету и загрузить данные с метками или данные без меток*'
    )
    name = st.text_input('Название')
    description = st.text_area('Описание')

    st.markdown('*Данные с метками*')
    left_column, right_column = st.beta_columns(2)
    dataset = left_column.file_uploader('Архив с обучающими данными')
    test_dataset = right_column.file_uploader('Архив с тестовыми данными')

    st.markdown('*Данные без меток*')
    unlabelled_dataset = st.file_uploader('Архив с данными без разметки')

    save_button = st.button('Сохранить')

    if save_button:
        if _save_dataset(name, description, dataset, test_dataset, unlabelled_dataset):
            st.info('Сохранено')
            time.sleep(1)
            session_state.subpage = None
            st.experimental_rerun()


def _save_dataset(name, description, dataset, test_dataset, unlabelled_dataset):
    if not name:
        st.error('Необходимо задать название датасету')
        return False
    if dataset and not test_dataset:
        st.error('Необходимо загрузить тестовые данные')
        return False
    if not dataset and not test_dataset and not unlabelled_dataset:
        st.error('Необходимо загрузить данные с метками или без меток')
        return False

    with create_session() as session:
        dataset = db.Dataset(name=name, description=description)  # noqa
        session.add(dataset)
    return True


def edit_dataset():
    st.markdown(
        """
        Можно данные в существующий датасет
        - выбор датасета
        - траин или тест?
        - данные в архив? csv?
        - кнопка, есть метка или нет?
    """
    )


def show_dataset():
    st.dataframe(
        get_dataframe_from_query(
            Query(
                [
                    db.Dataset.id,
                    db.Dataset.name.label('Название датасета'),
                    db.Dataset.description.label('Описание'),
                    db.Dataset.created_at.label('Дата создания'),
                    db.Dataset.updated_at.label('Дата последнего изменения'),
                ]
            )
        )
    )

    st.markdown(
        """
        Показываем таблицу существующих датасетов (из базы)
        - Название датасета
        - Описание датасета
        - Процент размеченных данных в датасете
        - Классы в датасете
        - ?? Версия датасета (время создания и время последнего обновления)
        
        Показываем сами картинки!
    """
    )
