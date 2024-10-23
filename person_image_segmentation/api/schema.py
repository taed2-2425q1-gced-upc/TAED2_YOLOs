"""
This module defines the data models used for responses in a prediction and energy tracking API.
"""

from typing import Optional
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """Represents the response for a prediction.

    Attributes:
        filename (str): The name of the file for which the prediction was made.
        message (str): A message describing the prediction result or any additional 
                       information.
    """
    filename: str
    message: str

class ErrorResponse(BaseModel):
    """Represents an error response.

    Attributes:
        error (str): A description of the error that occurred.
    """
    error: str

class RootResponse(BaseModel):
    """Represents the response for the root endpoint.

    Attributes:
        message (str): A welcome or informational message.
    """
    message: str

class EnergyStats(BaseModel):
    """Represents energy consumption statistics.

    Attributes:
        emissions (Optional[float]): The amount of emissions produced (e.g., CO2) during the 
            process.
        duration (Optional[float]): The duration of the process in seconds.
        cpu_power (Optional[float]): The power consumed by the CPU during the process in watts.
        gpu_power (Optional[float]): The power consumed by the GPU during the process in watts.
        ram_power (Optional[float]): The power consumed by the RAM during the process in watts.
        energy_consumed (Optional[float]): The total energy consumed during the process in 
            kilowatt-hours (kWh).
    """
    emissions: Optional[float] = None
    duration: Optional[float] = None
    cpu_power: Optional[float] = None
    gpu_power: Optional[float] = None
    ram_power: Optional[float] = None
    energy_consumed: Optional[float] = None

class PredictionAndEnergyResponse(BaseModel):
    """Represents the response that includes both prediction details and energy statistics.

    Attributes:
        prediction (PredictionResponse): The details of the prediction made.
        energy_stats (EnergyStats): The statistics related to energy consumption during the 
                                    prediction.
        message (str): A message providing additional information about the response.
    """
    prediction: PredictionResponse
    energy_stats: EnergyStats
    message: str
