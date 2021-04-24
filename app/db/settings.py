import sqlalchemy as sa
from pydantic import AnyUrl, BaseSettings


class DBSettings(BaseSettings):
    url: AnyUrl = 'postgresql://postgres:postgres@localhost:15432/kabanchiki'
    echo: int = 1

    class Config:
        env_prefix = 'DB_'

    @property
    def engine(self) -> sa.engine.Engine:
        return sa.create_engine(str(self.url), echo=self.echo)

    def setup_db(self) -> None:
        from app.db.base import metadata

        metadata.bind = self.engine
