import streamlit as st

from app.pages.utils import Page, render_horizontal_pages
from app.state import session_state


def app():
    pages = [
        Page('## Создать новый проект', create_new_project),
        Page('## Просмотр проектов', show_projects),
    ]
    render_horizontal_pages(pages, session_state=session_state)


def create_new_project():
    st.markdown(
        """
        Можно создать новый проект
        Задаем: Название + Описание (
                данные, модели, отчеты,
                метрики будут дальше)
    """
    )


def show_projects():
    st.markdown(
        """
        Показываем таблицу существующий (из базы)
        - Название проекта
        - Описание проекта
        - Модели/эксперименты или лучшая
        - Параметры модели
        - Лучшая метрика
        - Отчет
    """
    )
