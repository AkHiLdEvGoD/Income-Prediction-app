import pandas as pd
from api.schemas.predict_schema import InputData
from fastapi.responses import JSONResponse
from fastapi import APIRouter,Request,HTTPException
from api.utils.feature_engineering import feature_engineering

router = APIRouter()

@router.post('/predict')
async def predict(request:Request,input_data:InputData):
    try:
        input_df = pd.DataFrame([input_data.dict(by_alias=True)])
        model = request.app.state.model
        preprocessor = request.app.state.preprocessor
        
        input_df = feature_engineering(input_df)
        transformed = preprocessor.transform(input_df)
        pred = model.predict(transformed)
        if pred[0] == 0:
            return JSONResponse(status_code=200, content={'Predicted Income': '<=50k'})
        else:
            return JSONResponse(status_code=200, content={'Predicted Income': '>50k'})

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Prediction error: {e}")