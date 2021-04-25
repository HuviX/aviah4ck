import sqlalchemy as sa
from sqlalchemy.orm import relationship

from app.db.base import Base


class Project(Base):
    __tablename__ = 'project'

    name = sa.Column(sa.Text, nullable=False, unique=True)
    description = sa.Column(sa.Text, nullable=True)
    model = relationship('Model', backref='project', cascade='all, delete')
