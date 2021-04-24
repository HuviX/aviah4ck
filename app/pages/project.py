import time

import streamlit as st
from sqlalchemy import case, desc
from sqlalchemy.orm import Query

from app import db
from app.db.utils import create_session, get_dataframe_from_query
from app.pages.utils import Page, render_horizontal_pages
from app.state import session_state


def app():
    st.markdown(
        """
    *Проект - это конкретная задача, которая потребует множество экспериментов с данными и моделями*
    """
    )
    pages = [
        Page('## Просмотр проектов', show_projects),
        Page('## Создать новый проект', create_new_project),
    ]
    render_horizontal_pages(pages, session_state=session_state)


def create_new_project():
    st.markdown(
        f"""
            - *Необходимо дать название проекту*
            - *В описании проекта стоит указать цель исследования*
        """
    )
    name = st.text_input('Название')
    description = st.text_area('Описание')
    save_button = st.button('Сохранить')
    if save_button:
        if _save_project(name, description):
            st.info('Сохранено')
            time.sleep(1)
            session_state.subpage = None
            st.experimental_rerun()


def _save_project(name, description):
    if not name:
        st.error('Необходимо задать название проекту')
        return False

    with create_session() as session:
        session.add(db.Project(name=name, description=description))

    return True


def show_projects():
    df = get_dataframe_from_query(
        Query(
            [
                db.Project.name.label('Название проекта'),
                db.Project.description.label('Описание'),
            ]
        )
    )
    st.dataframe(df)

    selected_project = st.selectbox('Выбор проекта', df['Название проекта'].unique())
    st.dataframe(
        get_dataframe_from_query(
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
            .filter(db.Project.name == selected_project)
            .order_by(desc(db.Model.created_at))
        )
    )

    st.markdown("""TODO Тут должен быть отчет""")
