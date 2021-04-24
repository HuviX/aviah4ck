import streamlit as st


def app():
    st.markdown("""
        ### Датасеты 

        Можно добавить свой датасет
        - название
        - описание
        - данные в архив? csv?
        - тестовые данные в архив? csv?
        - кнопка, есть метка или нет?
        
        Добавить данные в существующий датасет
        - выбор датасета
        - траин или тест?
        - данные в архив? csv?
        - кнопка, есть метка или нет?
        
        Показываем таблицу существующих датасетов (из базы)
        - Название датасета
        - Описание датасета
        - Процент размеченных данных в датасете
        - Классы в датасете
        - ?? Версия датасета (время создания и время последнего обновления)
    """)

