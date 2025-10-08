from __future__ import annotations
"""SQLAlchemy database setup for Postgres (or any SQLAlchemy URL).

Provides a session factory and Base declarative.
"""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, scoped_session


class Base(DeclarativeBase):
    pass

def get_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True)


def create_scoped_session(database_url: str) -> scoped_session:
    engine = get_engine(database_url)
    return scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
