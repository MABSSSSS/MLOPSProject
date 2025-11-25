import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# -----------------------------------------
# DOCKER-FRIENDLY POSTGRES CONNECTION
# -----------------------------------------
# "host.docker.internal" allows the Docker container to connect
# to your Windows machine's PostgreSQL.
# -----------------------------------------

SQLALCHEMY_DATABASE_URL = os.getenv(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+psycopg2://postgres:1234@db:5432/MLOPS"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
