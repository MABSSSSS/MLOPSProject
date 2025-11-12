from sqlalchemy import Column, Integer, Float, String, DateTime,ForeignKey 
from datetime import datetime
from src.api.database.database import Base
from sqlalchemy.orm import relationship 
from src.api.models.user_models import User 


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
    
    
    # foreign key 
    
    # user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="last_prediction",uselist=False)
