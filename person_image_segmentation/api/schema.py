from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str
    message: str

class ErrorResponse(BaseModel):
    error: str

class RootResponse(BaseModel):
    message: str