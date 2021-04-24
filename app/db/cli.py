import pandas as pd

from app import db
from app.db.base import Base
from app.db.utils import init_db, create_session

from app.db.settings import DBSettings


def main():
    settings = DBSettings()
    settings.setup_db()
    init_db(metadata=Base.metadata, db_url=settings.url, drop_existing=True)

    datasets = pd.read_csv('app/db/dummy/dataset.csv')
    with create_session() as session:
        for row in datasets.itertuples():
            dataset = db.Dataset(
                name=row.name,  # noqa
                description=row.description  # noqa
            )
            session.add(dataset)


if __name__ == '__main__':
    main()
