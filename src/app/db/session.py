"""Database connection and session management.

Provides engine initialization, session retrieval, and database schema creation.
Supports lazy loading of connection pools to facilitate runtime settings configuration.
"""

import logging
from typing import Any, Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.app.core.settings import settings

logger = logging.getLogger(__name__)

# Global singletons initialized lazily
_engine: Engine | None = None


class LazySessionmaker(sessionmaker[Session]):
    """A sessionmaker subclass that ensures get_engine() is called before instantiating sessions."""

    def __call__(self, *args: Any, **kwargs: Any) -> Session:  # noqa: ANN401
        """Create a new session, ensuring the database engine is initialized first.

        Args:
            *args: Variable length argument list passed to the superclass constructor.
            **kwargs: Arbitrary keyword arguments passed to the superclass constructor.

        Returns:
            Session: The constructed SQLAlchemy Session instance.
        """
        # Ensure engine is initialized and SessionLocal is configured
        get_engine()
        return super().__call__(*args, **kwargs)


SessionLocal = LazySessionmaker(autocommit=False, autoflush=False)


def get_engine() -> Engine:
    """Retrieve or construct the database engine.

    Returns:
        Engine: The SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is None:
        db_url = settings.database_url or ""
        if db_url:
            from sqlalchemy.engine import make_url
            url_obj = make_url(db_url)
            redacted_url = url_obj.render_as_string(hide_password=True)
            logger.info("Initializing database engine with URL: %s", redacted_url)
        else:
            logger.info("Initializing database engine with empty URL")
        if db_url.startswith("sqlite"):
            from sqlalchemy.pool import StaticPool
            _engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            _engine = create_engine(db_url)
        SessionLocal.configure(bind=_engine)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Retrieve or construct the local session factory.

    Returns:
        sessionmaker: The SQLAlchemy sessionmaker.
    """
    # Trigger engine initialization to configure SessionLocal
    get_engine()
    return SessionLocal


def init_db() -> None:
    """Initialize all schema tables defined in the Base metadata."""
    from src.app.db.models import Base

    engine = get_engine()
    logger.info("Creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session context.

    Yields:
        Session: The database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

