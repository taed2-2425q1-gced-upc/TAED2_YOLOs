from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    filename: str
    message: str

class ErrorResponse(BaseModel):
    error: str

class RootResponse(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    filename: str
    message: str

class EnergyStats(BaseModel):
    emissions: Optional[float] = None
    duration: Optional[float] = None
    cpu_power: Optional[float] = None
    gpu_power: Optional[float] = None
    ram_power: Optional[float] = None
    energy_consumed: Optional[float] = None

class PredictionAndEnergyResponse(BaseModel):
    prediction: PredictionResponse
    energy_stats: EnergyStats
    message: str
