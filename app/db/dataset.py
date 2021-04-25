from enum import Enum

import sqlalchemy as sa

from app.db.base import Base


class DatasetType(Enum):
    TRAIN = 'train'
    TEST = 'test'
    UNLABELLED = 'unlabelled'


class Dataset(Base):
    __tablename__ = 'dataset'

    name = sa.Column(sa.String, nullable=False, unique=True)
    description = sa.Column(sa.Text, nullable=True)
    train_count = sa.Column(sa.Integer, nullable=False, default=0)
    test_count = sa.Column(sa.Integer, nullable=False, default=0)
    unlabelled_count = sa.Column(sa.Integer, nullable=False, default=0)
