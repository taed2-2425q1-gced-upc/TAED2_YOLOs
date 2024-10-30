"""
This module contains tests for the YOLO-based image segmentation API, including
prediction functionality, token-based authentication, and automated cleanup of
old files.

Tests include:
- Verification of authentication and authorization.
- Mask prediction on images using YOLO.
- Prediction with energy emissions tracking.
- Scheduled deletion of old files on the server.
"""

# pylint: disable=W0621
from pathlib import Path
import os
import time
import asyncio
import sys
from unittest.mock import patch
from http import HTTPStatus
import pytest
import pandas as pd

from fastapi.testclient import TestClient
from person_image_segmentation.api.app import app, clean_old_images, schedule_cleaning_task,lifespan
from person_image_segmentation.config import REPO_PATH, VALID_TOKEN


@pytest.fixture(scope="function")
def client():
    """
    Fixture that provides a test client for the API respecting the lifecycle (lifespan),
    allowing the loading of molds and cleaning tasks.
    """
    with TestClient(app) as client:
        yield client

@pytest.fixture
def payload():
    """
    Fixture that returns a payload with a JPEG test image and a valid authorization 
    token for mask prediction tests.

    Returns:
        dict: Dictionary with the image and authorization headers.
    """
    path = Path(REPO_PATH) / "tests/test_images/test_image.jpg"
    if not path.exists():
        pytest.fail(f"Test image not found at {path}")

    return {
        "file": ("test_image.jpg", open(path, "rb"), "image/jpeg"),
        "headers": {"Authorization": f"Bearer {VALID_TOKEN}"},
    }

@pytest.fixture
def non_jpeg_payload():
    """
    Fixture that returns a payload with a PNG test image and a valid authorization 
    token for mask prediction tests.

    Returns:
        dict: Dictionary with the PNG image and authorization headers.
    """
    path = Path(REPO_PATH) / "tests/test_images/test_image.png"
    if not path.exists():
        pytest.fail(f"Test non-JPEG image not found at {path}")

    return {
        "file": ("test_image.png", open(path, "rb"), "image/png"),
        "headers": {"Authorization": f"Bearer {VALID_TOKEN}"},
    }

@pytest.fixture
def non_mask_payload():
    """
    Fixture that returns a payload with a png test image that has no persons in it
    (should not return any masks) and a valid authorization token for mask 
    prediction tests.

    Returns:
        dict: Dictionary with the PNG image and authorization headers.
    """
    path = Path(REPO_PATH) / "tests/test_images/test_image_no_mask.png"
    if not path.exists():
        pytest.fail(f"Test non-mask image not found at {path}")

    return {
        "file": ("test_image_no_mask.png", open(path, "rb"), "image/png"),
        "headers": {"Authorization": f"Bearer {VALID_TOKEN}"},
    }

def test_favicon(client):
    """
    Test that the favicon route returns the file correctly.
    """
    response = client.get("/favicon.ico")
    assert response.status_code == HTTPStatus.OK, "Favicon endpoint failed"
    assert response.headers["content-type"] == "image/x-icon"

def test_read_root(client):
    """
    Tests the root endpoint to verify that the API responds correctly and returns 
    a welcome message.
    """
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": "API para hacer predicciones con YOLO"}

@patch.dict(sys.modules, {'torch': None})
def test_predict_mask_function_without_module(client, payload):
    """
    Tests the predict_mask_function when the torch module is not installed.
    Verifies that the function raises an INTERNAL_SERVER_ERROR.
    """
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers=payload["headers"],
        )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

def test_predict_mask_with_valid_token(client, payload):
    """
    Tests mask prediction on a valid JPEG image with a valid authorization token.
    Verifies that the response includes a filename and a message indicating that
    the prediction is complete.
    """
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers=payload["headers"],
        )

    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert "filename" in response_json
    assert response_json["message"] == "Prediction complete!"

def test_predict_mask_with_invalid_token(client, payload):
    """
    Tests mask prediction with an invalid authorization token. Verifies that the 
    API responds with a 401 authorization error.
    """
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers={"Authorization": "Bearer INVALID_TOKEN"},
        )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "Token inválido o no autorizado"

def test_predict_mask_with_no_file(client, payload):
    """
    Tests the prediction endpoint without sending an image file. Verifies that 
    the API responds with a validation error (422) due to the missing file.
    """
    response = client.post(
        "/predict/image/",
        headers=payload["headers"],
    )

    # As there is no HTTPStatus attribute for this code, we use the code directly
    assert response.status_code == 422

def test_predict_mask_with_non_jpeg_file(client, non_jpeg_payload):
    """
    Tests mask prediction on a valid PNG image with a valid authorization token. 
    Verifies that the response includes a filename and a message indicating that
    the prediction is complete.
    """
    with open(non_jpeg_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/",
            files={"file": ("test_image.png", image_file, "image/png")},
            headers=non_jpeg_payload["headers"],
        )

    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert "filename" in response_json
    assert response_json["message"] == "Prediction complete!"

def test_predict_mask_with_no_masks(client, non_mask_payload):
    """
    Tests mask prediction on an image without detectable masks. Verifies that the 
    API responds with a 500 error and a message indicating no masks were found 
    in the prediction.
    """
    with open(non_mask_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/",
            files={"file": ("test_image_no_mask.png", image_file, "image/png")},
            headers=non_mask_payload["headers"],
        )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = response.json()
    assert "No masks found in the prediction." in response_json["detail"]

def test_predict_mask_with_non_jpeg_file_with_csv(client, non_jpeg_payload):
    """
    Tests mask prediction on a valid PNG image and verifies that the energy emissions 
    CSV file is created in the file system.
    """
    emissions_path = Path(REPO_PATH) / "static" / "emissions_inference_api.csv"
    emissions_path.parent.mkdir(parents=True, exist_ok=True)
    expected_data = pd.DataFrame([{
        'emissions': 0.1,
        'duration': 2,
        'cpu_power': 15,
        'gpu_power': 25,
        'ram_power': 10,
        'energy_consumed': 1.5,
    }])
    expected_data.to_csv(emissions_path, index=False)

    with open(non_jpeg_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/emissions/",
            files={"file": ("test_image.png", image_file, "image/png")},
            headers=non_jpeg_payload["headers"],
        )

    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert "filename" in response_json['prediction']
    assert response_json["message"] == "Prediction complete with energy tracking!"

def test_predict_with_emissions_with_valid_token(client, payload):
    """
    Tests mask prediction with energy tracking on a valid JPEG image using a 
    valid authorization token. Verifies that the response includes prediction data 
    and energy statistics.
    """
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/emissions/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers=payload["headers"],
        )

    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert "prediction" in response_json
    assert "energy_stats" in response_json
    assert response_json["message"] == "Prediction complete with energy tracking!"

def test_predict_with_emissions_with_invalid_token(client, payload):
    """
    Tests mask prediction (and emissions) with an invalid authorization token. Verifies that the 
    API responds with a 401 authorization error.
    """
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/emissions/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers={"Authorization": "Bearer INVALID_TOKEN"},
        )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "Token inválido o no autorizado"

def test_predict_emissions_mask_with_no_masks(client, non_mask_payload):
    """
    Tests mask prediction on an image without detectable masks. Verifies that the 
    API responds with a 500 error and a message indicating no masks were found 
    in the prediction.
    """
    with open(non_mask_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/emissions",
            files={"file": ("test_image_no_mask.png", image_file, "image/png")},
            headers=non_mask_payload["headers"],
        )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = response.json()
    assert "No masks found in the prediction." in response_json["detail"]

def test_predict_mask_with_emissions_with_no_existing_file(client, payload):
    """
    Tests the prediction endpoint without sending an image file. Verifies that 
    the API responds with a validation error (500) due to the missing file.
    """
    response = client.post(
        "/predict/image/emissions/",
        files={"file": ("invalid_file.txt", b"not an image")},
        headers=payload["headers"],
    )

    # As there is no HTTPStatus attribute for this code, we use the code directly
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

def test_predict_with_emissions_non_jpeg_image(client, non_jpeg_payload):
    """
    Tests mask prediction with energy tracking on a valid PNG image. Verifies 
    that the response includes prediction data and energy statistics.
    """
    with open(non_jpeg_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/image/emissions/",
            files={"file": ("test_image.png", image_file, "image/png")},
            headers=non_jpeg_payload["headers"],
        )

    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert "prediction" in response_json
    assert "energy_stats" in response_json
    assert response_json["message"] == "Prediction complete with energy tracking!"

def test_clean_old_images():
    """
    Tests the `clean_old_images` function by creating an old test file and 
    verifying that the function successfully deletes it.
    """
    # Create file older than 10 minutes in static folder
    old_file_path = Path(REPO_PATH) / "static" / "old_test_image.jpg"
    old_file_path.touch()
    os.utime(old_file_path, (time.time() - 601, time.time() - 601))

    clean_old_images()

    # Verify that the file was deleted
    assert not old_file_path.exists(), "La función `clean_old_images` no eliminó el archivo."

@pytest.mark.anyio
async def test_schedule_cleaning_task():
    """
    Tests the scheduled cleanup task `schedule_cleaning_task`. Starts the cleanup 
    task and cancels it after a short period to verify that the task starts correctly.
    """
    # Create the cleanup task and cancel it after a short time
    cleaning_task = asyncio.create_task(schedule_cleaning_task())
    await asyncio.sleep(1)
    cleaning_task.cancel()

    try:
        await cleaning_task
    except asyncio.CancelledError:
        pass


@pytest.mark.anyio
async def test_lifespan():
    """
    Test the `lifespan` lifecycle to ensure it starts and stops correctly 
    the scheduled cleaning task.
    """
    old_file_path = Path(REPO_PATH) / "static" / "lifespan_old_image.jpg"
    old_file_path.touch()
    os.utime(old_file_path, (time.time() - 601, time.time() - 601))

    async with lifespan(app):
        await asyncio.sleep(5)  # Wait long enough for the cleanup task to run

    assert not old_file_path.exists(), "No eliminó el archivo en el ciclo de vida."
