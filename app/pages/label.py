from pathlib import Path

import streamlit as st
from sqlalchemy.orm import Query
from streamlit_labelstudio import st_labelstudio

from app import db
from app.db.utils import get_dataframe_from_query


def app():
    st.markdown(
        """
    *Редактор позволяет размечать данные*
    *Возможности редактора:*
    - Приближение/Удаление и позиционирование фото
    - Выделение проблемных мест на деталях с классом дефекта
    - Сохраняется история разметки
    """
    )

    df = get_dataframe_from_query(Query([db.Dataset.id, db.Dataset.name]))
    selected_dataset = st.selectbox('Выбор датасета', df.name.unique())
    dataset_id = df[df.name == selected_dataset].id.values[0]

    photo_paths = list((Path('data') / str(dataset_id) / 'unlabelled').rglob('*.png'))
    selected_photo = st.selectbox('Фотография без метки', photo_paths)

    if selected_photo:
        tasks = {
            'completions': [],
            'predictions': [],
            'id': 1,
            'data': {'image': f'http://localhost:5000/{selected_photo}'},
        }

        results_raw = _get_labelstudio_labels(tasks)

        if results_raw is not None:
            areas = [v for k, v in results_raw['areas'].items()]

            results = []
            for a in areas:
                results.append(
                    {
                        'id': a['id'],
                        'x': a['x'],
                        'y': a['y'],
                        'width': a['width'],
                        'height': a['height'],
                        'label': a['results'][0]['value']['rectanglelabels'][0],
                    }
                )

            st.table(results)


def _get_labelstudio_labels(tasks):
    config = """
      <View>
        <RectangleLabels name="tag" toName="img">
          <Label value="Дефект"></Label>
        </RectangleLabels>
        <View style="padding: 25px; box-shadow: 2px 2px 8px #AAA;">
          <Image name="img" value="$image" brightnessControl="true" contrastControl="true" zoomControl="true"></Image>
        </View>
      </View>
    """

    interfaces = (['panel', 'update', 'controls'],)

    user = {'pk': 1, 'firstName': 'James', 'lastName': 'Dean'}

    return st_labelstudio(config, interfaces, user, tasks)
