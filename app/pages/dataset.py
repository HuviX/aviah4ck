import streamlit as st

from app.pages.utils import Page, render_horizontal_pages


def app():
    pages = [
        Page('## Создать новый датасет', create_new_dataset),
        Page('## Добавить данные в существующий датасет', add_or_remove_examples),
        Page('## Просмотр датасетов', show_datasets),
    ]
    render_horizontal_pages(pages)


def create_new_dataset():
    st.markdown(
        """
        Можно добавить свой датасет
        - название
        - описание
        - данные в архив? csv?
        - тестовые данные в архив? csv?
        - кнопка, есть метка или нет?
    """
    )


def add_or_remove_examples():
    st.markdown(
        """
        Можно данные в существующий датасет
        - выбор датасета
        - траин или тест?
        - данные в архив? csv?
        - кнопка, есть метка или нет?
    """
    )


def show_datasets():
    st.markdown(
        """
        Показываем таблицу существующих датасетов (из базы)
        - Название датасета
        - Описание датасета
        - Процент размеченных данных в датасете
        - Классы в датасете
        - ?? Версия датасета (время создания и время последнего обновления)
    """
    )
