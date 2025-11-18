from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# -----------------------------------------
# DATABASE CONNECTION STRING
# -----------------------------------------
# Format:
# postgresql+psycopg2://<username>:<password>@<host>:<port>/<database_name>
#
# This is the connection URL for PostgreSQL.
# SQLAlchemy uses this URL to connect to your database.
# -----------------------------------------
SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/MLOPS"

# -----------------------------------------
# DATABASE ENGINE
# -----------------------------------------
# The engine is the "core" connection to the database.
# All SQL queries issued by SQLAlchemy are executed through this engine.
# -----------------------------------------
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# -----------------------------------------
# SESSION LOCAL
# -----------------------------------------
# A session is used to interact with the database (CRUD operations).
# autocommit=False → You control when to commit
# autoflush=False  → Prevents accidental writes
# bind=engine      → This session is connected to the above engine
# -----------------------------------------
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -----------------------------------------
# BASE CLASS FOR ORM MODELS
# -----------------------------------------
# Every SQLAlchemy ORM model will inherit from this Base class.
# Example:
# class User(Base):
#     __tablename__ = "users"
# -----------------------------------------
Base = declarative_base()

# -----------------------------------------
# FASTAPI DEPENDENCY
# -----------------------------------------
# Provides a new database session for each request.
# Ensures:
# - Session is created when needed
# - Session is closed after request (avoid memory leaks)
#
# Usage in routes:
# def get_users(db: Session = Depends(get_db)):
# -----------------------------------------
def get_db():
    db = SessionLocal()      # create session
    try:
        yield db             # provide session to route
    finally:
        db.close()           # close session after request
