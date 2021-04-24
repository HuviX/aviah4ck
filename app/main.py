from typing import Any, Dict

import streamlit as st

from app.pages import (
    about_app,
    dataset_app,
    head_app,
    inference_app,
    label_app,
    project_app,
    train_app,
)

TITLE = 'Кабанчики-ML'
LOGO_URL = 'img/logo.png'
LOGO_AVIAHACK_URL = 'img/logo_aviahack.png'


def main():
    st.sidebar.image(LOGO_URL)
    st.sidebar.markdown(
        f"""
            # {TITLE}
            [![Star](https://img.shields.io/github/stars/HuviX/aviah4ck.svg?logo=github&style=social)](https://gitHub.com/HuviX/aviah4ck)
        """
    )
    pages = [
        {'name': 'Главная страница', 'app': head_app},
        {'name': 'Датасеты', 'app': dataset_app},
        {'name': 'Проекты', 'app': project_app},
        {'name': 'Разметка', 'app': label_app},
        {'name': 'Обучение', 'app': train_app},
        {'name': 'Предсказание', 'app': inference_app},
        {'name': 'Об инструменте', 'app': about_app},
        {'name': 'Описание задачи', 'app': lambda: st.markdown(get_readme())},
    ]
    for page in pages:
        if st.sidebar.button(page['name']):
            render_page(page)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.image(LOGO_AVIAHACK_URL)


def get_readme():
    with open('README.md') as f:
        text = f.read()
    return text


def render_page(page: Dict[str, Any]):
    st.title(page['name'])
    page['app']()


if __name__ == '__main__':
    st.set_page_config(
        page_title=TITLE, page_icon=LOGO_URL, layout='wide',
    )
    main()
