from pydantic import BaseModel,Field 

class HouseData(BaseModel):
    GrLivArea: float
    OverallQual: int
    GarageCars: int
    YearBuilt: int
    TotalBsmtSF: float
    FullBath: int
    FirstFlrSF: float
    MSZoning: str
    Exterior1st: str
    Exterior2nd: str
    BsmtQual: str
    Foundation: str
    ExterQual: str
    HouseStyle: str
