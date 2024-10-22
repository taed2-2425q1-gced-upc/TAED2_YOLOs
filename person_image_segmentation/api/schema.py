from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str
    message: str

class PredictionAndEnergyResponse(BaseModel):
    prediction: PredictionResponse
    energy_stats: dict = {
        "json_schema_extra": {
            "example": {
                    'emissions': 10.3,
                    'duration': 10.3,
                    'cpu_power': 10.3,
                    'gpu_power': 10.3,
                    'ram_power': 10.3,
                    'energy_consumed': 10.3
            }
        }
    }
    message: str

class ErrorResponse(BaseModel):
    error: str

class RootResponse(BaseModel):
    message: str