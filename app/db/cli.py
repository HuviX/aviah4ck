import pandas as pd

from app import db
from app.db.base import Base
from app.db.settings import DBSettings
from app.db.utils import create_session, init_db


def main():
    settings = DBSettings()
    settings.setup_db()
    init_db(metadata=Base.metadata, db_url=settings.url, drop_existing=True)

    with create_session() as session:
        for row in pd.read_csv('app/db/dummy/dataset.csv').itertuples():
            session.add(
                db.Dataset(
                    **{k: v for k, v in row._asdict().items() if k != 'Index'}  # noqa
                )
            )

    with create_session() as session:
        for row in pd.read_csv('app/db/dummy/project.csv').itertuples():
            session.add(
                db.Project(
                    **{k: v for k, v in row._asdict().items() if k != 'Index'}  # noqa
                )
            )

    with create_session() as session:
        for row in pd.read_csv('app/db/dummy/model.csv').itertuples():
            session.add(
                db.Model(
                    **{k: v for k, v in row._asdict().items() if k != 'Index'}  # noqa
                )
            )


if __name__ == '__main__':
    main()
