from fastapi import FastAPI
from .utils.model_loader import lifespan
from .routes.predict import router as predict_router

app = FastAPI(title='Income Prediction API',lifespan=lifespan)
app.include_router(predict_router)

if __name__ == "__main__":
    main()