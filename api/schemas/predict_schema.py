from pydantic import BaseModel,Field
from typing import Annotated,Literal

class InputData(BaseModel):
    age: Annotated[int, Field(..., ge=0, le=120, description="Age of the individual (0–120)")]
    workclass: Annotated[str, Field(..., min_length=2, description="Type of work class")]
    education: Annotated[str, Field(..., min_length=2, description="Highest level of education")]
    educational_num: Annotated[int, Field(..., ge=1, le=20, description="Education level in numeric form (1–20)",alias = 'educational-num')]
    marital_status: Annotated[str, Field(..., min_length=2, description="Marital status",alias='marital-status')]
    occupation: Annotated[str, Field(..., min_length=2, description="Occupation type")]
    relationship: Annotated[str, Field(..., min_length=2, description="Relationship category")]
    race: Annotated[str, Field(..., min_length=2, description="Race of individual")]
    gender: Annotated[Literal["Male", "Female"], Field(..., description="Gender (Male or Female)")]
    capital_gain: Annotated[int, Field(..., ge=0, description="Capital gain (non-negative)",alias='capital-gain') ]
    capital_loss: Annotated[int, Field(..., ge=0, description="Capital loss (non-negative)",alias = 'capital-loss')]
    hours_per_week: Annotated[int, Field(..., ge=1, le=100, description="Number of working hours per week (1–100)",alias ='hours-per-week')]
    native_country: Annotated[str, Field(..., min_length=2, description="Country of origin",alias = 'native-country')]