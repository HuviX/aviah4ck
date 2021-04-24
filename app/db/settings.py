import sqlalchemy as sa

from pydantic import BaseSettings, AnyUrl


class DBSettings(BaseSettings):
    url: AnyUrl = 'postgresql://postgres:postgres@localhost:15432/kabanchiki'

    class Config:
        env_prefix = 'DB_'

    @property
    def engine(self) -> sa.engine.Engine:
        return sa.create_engine(str(self.url))

    def setup_db(self) -> None:
        from app.db.base import metadata

        metadata.bind = self.engine
