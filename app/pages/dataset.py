import logging
import tarfile
import time
from pathlib import Path

import streamlit as st
from sqlalchemy import desc
from sqlalchemy.orm import Query

from app import db
from app.db.dataset import DatasetType
from app.db.utils import create_session, get_dataframe_from_query
from app.pages.utils import Page, render_horizontal_pages
from app.state import session_state

logger = logging.getLogger(__name__)


def app():
    st.markdown(
        """
    *В датасетах хранятся данные 3 типов*
    - Обучающие данные
    - Тестовые данные
    - Сырые данные / данные без меток
    """
    )
    pages = [
        Page('## Просмотр датасета', show_dataset),
        Page('## Загрузить датасет', create_dataset),
        Page('## Редактировать датасет', edit_dataset),
    ]
    render_horizontal_pages(pages, session_state=session_state)


def create_dataset():
    st.markdown(
        f"""
        - *Необходимо дать название датасету и загрузить данные с метками или данные без меток*
        - *Данные в архиве формата `.tar`*
        - *Формат данных c метками*
        ```
        ├── {DatasetType.TRAIN}
        │   ├── img0.png
        │   ├── img1.png
        │   ├── img2.png
        │   └── labels.csv
        └── {DatasetType.TEST}
            ├── img3.png
            ├── img4.png
            ├── img5.png
            └── labels.csv
        ```
        - *Формат файла `labels.csv`*
        ```
        image	x	y	width	height
        0.png	70	19	100	    111
        1.png	108	91	89	    82
        2.png	90	19	115	    181
        ```
        - *Формат данных без меток*
        ```
        └── {DatasetType.UNLABELLED}
            ├── img0.png
            ├── img1.png
            └── img2.png
        ```
    """
    )
    name = st.text_input('Название')
    description = st.text_area('Описание')
    dataset = st.file_uploader('Архив c данными', type=['tar', 'tar.gz', 'tar.xz'])

    save_button = st.button('Сохранить')
    if save_button:
        if _save_dataset(name, description, dataset):
            st.info('Сохранено')
            time.sleep(1)
            session_state.subpage = None
            st.experimental_rerun()


def _save_dataset(name, description, dataset):
    if not name:
        st.error('Необходимо задать название датасету')
        return False

    if not dataset:
        st.error('Необходимо загрузить данные с метками или без меток')
        return False

    with create_session() as session:
        record = db.Dataset(name=name, description=description)  # noqa
        session.add(record)
        session.flush()

        try:
            uploaded_file = dataset
            path = Path(f'data/{record.id}/temp.tar.xz')
            path.parent.mkdir(exist_ok=True)

            with open(path, 'wb') as f:
                f.write(uploaded_file.read())

            with tarfile.open(path) as f:
                f.extractall(path=f'data/{record.id}/')

            record.train_count = len(
                list((path.parent / DatasetType.TRAIN.value).glob('*.png'))
            )
            record.test_count = len(
                list((path.parent / DatasetType.TEST.value).glob('*.png'))
            )
            record.unlabelled_count = len(
                list((path.parent / DatasetType.UNLABELLED.value).glob('*.png'))
            )

        except Exception as e:
            st.exception(e)
            return False

    return True


def edit_dataset():
    st.markdown(
        """
        TODO
        Можно данные в существующий датасет
        - выбор датасета
        - куда траин или тест или анлейбл?
        - по одной или несколько фоток?
        - кнопка, есть метка или нет?
    """
    )


def show_dataset():
    df = get_dataframe_from_query(
        Query(
            [
                db.Dataset.id,
                db.Dataset.name.label('Название датасета'),
                db.Dataset.description.label('Описание'),
                db.Dataset.train_count.label('Обучающих'),
                db.Dataset.test_count.label('Тестовых'),
                db.Dataset.unlabelled_count.label('Сырых'),
                db.Dataset.created_at.label('Дата создания'),
                db.Dataset.updated_at.label('Последнее изменение'),
            ]
        )
        .order_by(desc(db.Dataset.updated_at))
        .limit(50)
    )
    st.dataframe(df.drop('id', axis=1))

    selected_dataset = st.selectbox('Выбор датасета', df['Название датасета'].unique())

    with create_session() as session:
        dataset_id = (
            session.query(db.Dataset.id)
            .filter(db.Dataset.name == selected_dataset)
            .first()[0]
        )

    try:
        files = list((Path('data') / str(dataset_id)).rglob('*.png'))
        selected_photo = st.selectbox('Фотография', files)
        st.image(str(selected_photo))
    except:
        pass
