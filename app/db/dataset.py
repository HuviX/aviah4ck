import sqlalchemy as sa

from app.db.base import Base


class Dataset(Base):
    __tablename__ = 'dataset'

    name = sa.Column(sa.Text, nullable=False)
    description = sa.Column(sa.Text, nullable=True)
