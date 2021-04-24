import sqlalchemy as sa

PK_TYPE = sa.Integer()


class Base:
    id = sa.Column(PK_TYPE, primary_key=True)
    created_at = sa.Column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )
    updated_at = sa.Column(
        sa.DateTime(timezone=True), onupdate=sa.func.now(), default=sa.func.now()
    )
