from fastapi import FastAPI
from src.api.routes.prediction import router as prediction_router
from fastapi.middleware.cors import CORSMiddleware 
from src.api.database.database import engine, Base
from src.api.models.prediction_models import HousePrediction
from src.api.routes import user_router 
# This will create all tables that inherit from Base
Base.metadata.create_all(bind=engine)

app = FastAPI(title="House Prediction API")

app.include_router(prediction_router, prefix="/api")
app.include_router(user_router.router,prefix="/users", tags=["Authentication"])

# Now root endpoint is available at /api/



# Optional: allow CORS if using frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)