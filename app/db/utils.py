import logging
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy.orm import Session as SessionClass
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists, drop_database

from app.db.settings import DBSettings

logger = logging.getLogger(__name__)


def init_db(metadata: MetaData, db_url: str, drop_existing: bool = False,) -> None:
    create = True

    if database_exists(db_url):
        if drop_existing:
            drop_database(db_url)
            logger.info('DB %s dropped', str(db_url))
        else:
            create = False

    if create:
        create_database(db_url)
        logger.info('DB %s created', str(db_url))
        metadata.create_all()

    logger.info('Database %s initialized', str(db_url))


Session = sessionmaker()


@contextmanager
def create_session(**kwargs: Any) -> Iterator[SessionClass]:
    new_session = Session(**kwargs)
    try:
        yield new_session
        new_session.commit()
    except Exception:
        new_session.rollback()
        raise
    finally:
        new_session.close()


def get_dataframe_from_query(query) -> pd.DataFrame:
    logger.info('Making database query')
    data = pd.read_sql(query.statement, DBSettings().url)
    logger.info(f'Got {len(data)} records')
    return data
