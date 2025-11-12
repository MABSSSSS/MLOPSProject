from sqlalchemy import Column, Integer, String, DateTime ,ForeignKey 

from datetime import datetime 
from sqlalchemy.orm import relationship 
from src.api.database.database import Base 

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True,index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    password = Column(String,nullable=False)
    created_at = Column(DateTime,default=datetime.utcnow)
    
    last_prediction_id = Column(Integer,ForeignKey("house_predictions.id"),nullable=True)
    
    
    last_prediction = relationship(
        "HousePrediction",
        uselist=False 
        
    )
    
    
    