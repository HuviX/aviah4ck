import streamlit as st


def app():
    st.markdown("""Тут что, шапка?""")
    st.markdown(
        """
    ### Описание инструмента
    Мол я нуб поясните куда кликать
    Тут инструкция как запустить и тд
    """
    )

    st.markdown('*Схема обработки пользовательских запросов*')
    st.image('img/client_seq.jpg')
