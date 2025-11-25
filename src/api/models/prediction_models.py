from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from datetime import datetime
from sqlalchemy.orm import relationship
from src.api.database.database import Base

class HousePrediction(Base):
    __tablename__ = "house_predictions"

    id = Column(Integer, primary_key=True, index=True)
    GrLivArea = Column(Float)
    OverallQual = Column(Integer)
    GarageCars = Column(Integer)
    YearBuilt = Column(Integer)
    TotalBsmtSF = Column(Float)
    FullBath = Column(Integer)
    FirstFlrSF = Column(Float)

    MSZoning = Column(String)
    Exterior1st = Column(String)
    Exterior2nd = Column(String)
    BsmtQual = Column(String)
    Foundation = Column(String)
    ExterQual = Column(String)
    HouseStyle = Column(String)

    predicted_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign key to User
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationship to User (1-to-many)
    user = relationship(
        "User",
        back_populates="predictions",
        foreign_keys=[user_id]  # ✅ Explicitly specify the foreign key
    )
