import sqlalchemy as sa
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import as_declarative

metadata = MetaData()

PK_TYPE = sa.Integer()


@as_declarative(metadata=metadata)
class Base:
    id = sa.Column(PK_TYPE, primary_key=True)
    created_at = sa.Column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )
    updated_at = sa.Column(
        sa.DateTime(timezone=True), onupdate=sa.func.now(), default=sa.func.now()
    )
