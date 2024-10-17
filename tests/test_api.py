import os
import pytest
from fastapi.testclient import TestClient
from person_image_segmentation.api.app import app

client = TestClient(app)
VALID_TOKEN = "YOLOs"

@pytest.fixture
def test_image_path():
    """
    Fixture for the path of a test image.
    """
    path = "/Users/mariarisques/TAED2_YOLOs/TAED2_YOLOs/tests/test_image.jpg"
    if not os.path.exists(path):
        pytest.fail(f"Test image not found at {path}")
    return path

@pytest.fixture
def test_non_jpeg_image_path():
    """
    Fixture for the path of a non-JPEG test image.
    """
    path = "/Users/mariarisques/TAED2_YOLOs/TAED2_YOLOs/tests/test_image.png"
    if not os.path.exists(path):
        pytest.fail(f"Test non-JPEG image not found at {path}")
    return path

def test_read_root():
    """
    Test the root endpoint to ensure the API is working correctly.
    """
    response = client.get("/")
    assert response.status_code == 200, "Root endpoint failed with unexpected status code."
    assert response.json() == {"message": "API para hacer predicciones con YOLO"}

def test_favicon():
    """
    Test that the favicon route returns the file correctly.
    """
    response = client.get("/favicon.ico")
    assert response.status_code == 200, "Favicon endpoint failed with unexpected status code."
    assert response.headers["content-type"] == "image/x-icon"

def test_predict_mask_with_valid_token(test_image_path):
    """
    Test the prediction route with a valid token and a JPEG image.
    """
    with open(test_image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        )

    assert response.status_code == 200, "Prediction failed with valid token."
    response_json = response.json()
    assert "filename" in response_json, "Response missing 'filename' field."
    assert response_json["message"] == "Prediction complete!"

def test_predict_mask_with_invalid_token(test_image_path):
    """
    Test the prediction route with an invalid token.
    """
    with open(test_image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers={"Authorization": "Bearer INVALID_TOKEN"},
        )

    assert response.status_code == 401, "API did not return 401 for invalid token."
    assert response.json()["detail"] == "Token inválido o no autorizado"

def test_predict_mask_with_no_file():
    """
    Test the prediction route without sending a file.
    """
    response = client.post(
        "/predict/",
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )

    assert response.status_code == 422, "API did not return 422 for missing file."

def test_predict_mask_with_non_jpeg_file(test_non_jpeg_image_path):
    """
    Test the prediction route with a non-JPEG image file.
    """
    with open(test_non_jpeg_image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.png", image_file, "image/png")},
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        )

    assert response.status_code == 200, "Prediction failed with non-JPEG file."
    response_json = response.json()
    assert "filename" in response_json, "Response missing 'filename' field for non-JPEG."
    assert response_json["message"] == "Prediction complete!"

def test_predict_mask_with_no_masks():
    """
    Test the prediction route when no masks are found in the prediction.
    """
    no_mask_image_path = "/Users/mariarisques/TAED2_YOLOs/TAED2_YOLOs/tests/test_image_no_mask.png"
    if not os.path.exists(no_mask_image_path):
        pytest.fail(f"Test image with no masks not found at {no_mask_image_path}")

    with open(no_mask_image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image_no_mask.png", image_file, "image/png")},
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        )

    # Verificar que la API devuelve un código de estado 400 cuando no se encuentran máscaras
    assert response.status_code == 400, "Expected status code 400 when no masks are found."
    response_json = response.json()
    assert "detail" in response_json, "Response should contain a 'detail' message when no masks are found."
    assert response_json["detail"] == "No masks found in the prediction.", "The error message should indicate that no masks were found."
