from collections import Callable
from dataclasses import dataclass
from typing import List

import streamlit as st


@dataclass
class Page:
    name: str
    app: Callable


def render_sidebar_pages(pages: List[Page], session_state):
    for page in pages:
        if st.sidebar.button(page.name.replace('#', '')):
            session_state.page = page
    render_page(session_state.page or pages[0])


def render_horizontal_pages(pages: List[Page], session_state=None):  # noqa
    lengths = [len(page.name) for page in pages]
    columns = st.beta_columns(lengths + lengths)  # 50/50
    buttons = [
        col.button(page.name.replace('#', '')) for col, page in zip(columns, pages)
    ]

    for button, page in zip(buttons, pages):
        if button:
            render_page(page)


def render_page(page: Page):
    st.markdown(page.name)
    page.app()
