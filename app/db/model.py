import sqlalchemy as sa

from app import db
from app.db.base import Base


class Model(Base):
    __tablename__ = 'model'

    name = sa.Column(sa.Text, nullable=False)
    description = sa.Column(sa.Text, nullable=True)
    dataset_id = sa.Column(
        sa.Integer, sa.ForeignKey(db.Dataset.id, ondelete='CASCADE'), nullable=True
    )
    params = sa.Column(sa.JSON, nullable=True)
    metrics = sa.Column(sa.JSON, nullable=True)
    training_time = sa.Column(sa.Integer, nullable=False, default=0)
    pretrained = sa.Column(sa.Boolean, default=False)

    project_id = sa.Column(sa.Integer, sa.ForeignKey('project.id'))
    path = sa.Column(sa.String, nullable=False)
