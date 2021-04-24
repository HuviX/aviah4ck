import SessionState
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
from app.pages.utils import Page, render_sidebar_pages
from app.db.settings import DBSettings

TITLE = 'Кабанчики-ML'
LOGO_URL = 'img/logo.png'
LOGO_AVIAHACK_URL = 'img/logo_aviahack.png'


session_state = SessionState.get(page=None)


def main():
    st.sidebar.image(LOGO_URL)
    st.sidebar.markdown(
        f"""
            # {TITLE}
            [![Star](https://img.shields.io/github/stars/HuviX/aviah4ck.svg?logo=github&style=social)](https://gitHub.com/HuviX/aviah4ck)
        """
    )
    pages = [
        Page('# Главная страница', head_app),
        Page('# Датасеты', dataset_app),
        Page('# Проекты', project_app),
        Page('# Разметка', label_app),
        Page('# Обучение', train_app),
        Page('# Предсказание', inference_app),
        Page('# Об инструменте', about_app),
        Page('# Описание задачи', lambda: st.markdown(get_readme())),
    ]
    render_sidebar_pages(pages, session_state=session_state)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.image(LOGO_AVIAHACK_URL)


def get_readme():
    with open('README.md') as f:
        text = f.read()
    return text


if __name__ == '__main__':
    DBSettings().setup_db()
    st.set_page_config(
        page_title=TITLE, page_icon=LOGO_URL, layout='wide',
    )
    main()
