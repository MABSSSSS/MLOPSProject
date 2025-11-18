from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from datetime import datetime
from sqlalchemy.orm import relationship
from src.api.database.database import Base
from src.api.models.user_models import User


class HousePrediction(Base):
    """
    ORM model for storing ML model predictions for house prices.
    This table will store:
    - All input features used during prediction
    - The predicted house price
    - Timestamp of prediction
    - (Optional) The user who made the prediction
    """
    
    __tablename__ = "house_predictions"  # Name of the table in PostgreSQL

    # -------------------------------
    # Primary Key (unique row ID)
    # -------------------------------
    id = Column(Integer, primary_key=True, index=True)

    # -------------------------------
    # Feature Columns (inputs to model)
    # Each column stores a user-supplied input
    # -------------------------------
    GrLivArea = Column(Float)
    OverallQual = Column(Integer)
    GarageCars = Column(Integer)
    YearBuilt = Column(Integer)
    TotalBsmtSF = Column(Float)
    FullBath = Column(Integer)
    FirstFlrSF = Column(Float)

    # Categorical columns
    MSZoning = Column(String)
    Exterior1st = Column(String)
    Exterior2nd = Column(String)
    BsmtQual = Column(String)
    Foundation = Column(String)
    ExterQual = Column(String)
    HouseStyle = Column(String)

    # -------------------------------
    # Predicted output
    # -------------------------------
    predicted_price = Column(Float)

    # -------------------------------
    # Timestamp of prediction
    # Automatically stores creation time
    # -------------------------------
    created_at = Column(DateTime, default=datetime.utcnow)

    # -------------------------------
    # Foreign Key → links prediction to a user
    # Allows storing "which user made this prediction"
    # -------------------------------
    # user_id = Column(Integer, ForeignKey("users.id"))

    # -------------------------------
    # Relationship with User Model
    # back_populates must match the attribute in User model
    #
    # Example in user_models.py:
    # predictions = relationship("HousePrediction", back_populates="user")
    #
    # This creates a ONE-TO-MANY relationship:
    #   One User → Many Predictions
    # -------------------------------
    user = relationship("User", back_populates="predictions")
