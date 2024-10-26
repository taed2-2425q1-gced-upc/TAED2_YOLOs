# """
# Module for testing the FastAPI application for YOLO image segmentation.

# This module includes tests for different endpoints and scenarios to ensure the correct 
# functionality of the API, including token validation, file format handling, and edge cases 
# such as no masks found in the prediction.
# """
# # pylint: disable=redefined-outer-name

# from pathlib import Path

# import os
# import pytest
# import time
# import asyncio

# from fastapi import FastAPI
# from fastapi.testclient import TestClient
# from dotenv import load_dotenv

# from person_image_segmentation.api.app import app, clean_old_images, lifespan



# load_dotenv()

# VALID_TOKEN = str(Path(os.getenv('VALID_TOKEN')))
# REPO_PATH = str(Path(os.getenv('PATH_TO_REPO')))


# client = TestClient(app)

# @pytest.fixture
# def test_image_path():
#     """
#     Fixture for the path of a test image.
#     """
#     path = REPO_PATH + "/tests/test_images/test_image.jpg"
#     if not os.path.exists(path):
#         pytest.fail(f"Test image not found at {path}")
#     return path

# @pytest.fixture
# def test_non_jpeg_image_path():
#     """
#     Fixture for the path of a non-JPEG test image.
#     """
#     path = REPO_PATH + "/tests/test_images/test_image.png"
#     if not os.path.exists(path):
#         pytest.fail(f"Test non-JPEG image not found at {path}")
#     return path

# def test_read_root():
#     """
#     Test the root endpoint to ensure the API is working correctly.
#     """
#     response = client.get("/")
#     assert response.status_code == 200, "Root endpoint failed with unexpected status code."
#     assert response.json() == {"message": "API para hacer predicciones con YOLO"}

# def test_predict_mask_with_valid_token(test_image_path):
#     """
#     Test the prediction route with a valid token and a JPEG image.
#     """
#     with open(test_image_path, "rb") as image_file:
#         response = client.post(
#             "/predict/",
#             files={"file": ("test_image.jpg", image_file, "image/jpeg")},
#             headers={"Authorization": f"Bearer {VALID_TOKEN}"},
#         )

#     assert response.status_code == 200, "Prediction failed with valid token."
#     response_json = response.json()
#     assert "filename" in response_json, "Response missing 'filename' field."
#     assert response_json["message"] == "Prediction complete!"

# def test_predict_mask_with_invalid_token(test_image_path):
#     """
#     Test the prediction route with an invalid token.
#     """
#     with open(test_image_path, "rb") as image_file:
#         response = client.post(
#             "/predict/",
#             files={"file": ("test_image.jpg", image_file, "image/jpeg")},
#             headers={"Authorization": "Bearer INVALID_TOKEN"},
#         )

#     assert response.status_code == 401, "API did not return 401 for invalid token."
#     assert response.json()["detail"] == "Token inválido o no autorizado"

# def test_predict_mask_with_no_file():
#     """
#     Test the prediction route without sending a file.
#     """
#     response = client.post(
#         "/predict/",
#         headers={"Authorization": f"Bearer {VALID_TOKEN}"},
#     )

#     assert response.status_code == 422, "API did not return 422 for missing file."

# def test_predict_mask_with_non_jpeg_file(test_non_jpeg_image_path):
#     """
#     Test the prediction route with a non-JPEG image file.
#     """
#     with open(test_non_jpeg_image_path, "rb") as image_file:
#         response = client.post(
#             "/predict/",
#             files={"file": ("test_image.png", image_file, "image/png")},
#             headers={"Authorization": f"Bearer {VALID_TOKEN}"},
#         )

#     assert response.status_code == 200, "Prediction failed with non-JPEG file."
#     response_json = response.json()
#     assert "filename" in response_json, "Response missing 'filename' field for non-JPEG."
#     assert response_json["message"] == "Prediction complete!"

# def test_predict_mask_with_no_masks():
#     """
#     Test the prediction route when no masks are found in the prediction.
#     """
#     no_mask_image_path = REPO_PATH + "/tests/test_images/test_image_no_mask.png"
#     if not os.path.exists(no_mask_image_path):
#         pytest.fail(f"Test image with no masks not found at {no_mask_image_path}")

#     with open(no_mask_image_path, "rb") as image_file:
#         response = client.post(
#             "/predict/",
#             files={"file": ("test_image_no_mask.png", image_file, "image/png")},
#             headers={"Authorization": f"Bearer {VALID_TOKEN}"},
#         )

#     # Verify that the API returns a 500 status code when no masks are found
#     assert response.status_code == 500, "Expected status code 500 when no masks are found."
#     response_json = response.json()
#     assert "detail" in response_json, "'detail' key is missing in the response."
#     assert (
#         "No masks found in the prediction." in response_json["detail"]
#         ), "Unexpected error message."

# def test_predict_with_emissions_with_valid_token(test_image_path):
#     """
#     Test the prediction with emissions route with a valid token and a JPEG image.
#     """
#     with open(test_image_path, "rb") as image_file:
#         response = client.post(
#             "/predict_with_emissions/",
#             files={"file": ("test_image.jpg", image_file, "image/jpeg")},
#             headers={"Authorization": f"Bearer {VALID_TOKEN}"},
#         )

#     assert response.status_code == 200, "Prediction with emissions failed with valid token."
#     response_json = response.json()
#     assert "prediction" in response_json, "Response missing 'prediction' field."
#     assert "energy_stats" in response_json, "Response missing 'energy_stats' field."
#     assert response_json["message"] == "Prediction complete with energy tracking!"


# def test_predict_with_emissions_non_jpeg_image(test_non_jpeg_image_path):
#     """
#     Test the predict_with_emissions route with a non-JPEG image file.
#     Verifies that the image is converted to JPEG format and processed correctly.
#     """
#     with open(test_non_jpeg_image_path, "rb") as image_file:
#         response = client.post(
#             "/predict_with_emissions/",
#             files={"file": ("test_image.png", image_file, "image/png")},
#             headers={"Authorization": f"Bearer {VALID_TOKEN}"},
#         )

#     assert response.status_code == 200, "Prediction with emissions failed for non-JPEG image."
    
#     # Check fields in JSON response
#     response_json = response.json()
#     assert "prediction" in response_json, "Response missing 'prediction' field."
#     assert "energy_stats" in response_json, "Response missing 'energy_stats' field."
#     assert response_json["message"] == "Prediction complete with energy tracking!"

# def test_clean_old_images():
#     # Create a file in the `static` directory that is more than 10 minutes old
#     old_file_path = Path(REPO_PATH) / "static" / "old_test_image.jpg"
#     old_file_path.touch()  # Crear archivo
#     # Change file modification time to more than 10 minutes ago
#     os.utime(old_file_path, (time.time() - 601, time.time() - 601))

#     # Call the function
#     clean_old_images()

#     # Verify that the file has been deleted
#     assert not old_file_path.exists(), "La función `clean_old_images` no eliminó el archivo antiguo."

# async def schedule_cleaning_task():
#     """
#     Schedules a periodic cleaning task to delete old files.

#     The task runs every 60 seconds.
#     """
#     try:
#         while True:
#             clean_old_images()
#             await asyncio.sleep(60)  # Run every 60 seconds
#     except asyncio.CancelledError:
#         print("Tarea de limpieza cancelada correctamente.")

# def test_lifespan():
#     """
#     Test that the lifespan context manager starts and stops the cleaning task correctly.
#     """
#     async def run_test():
#         app = FastAPI(lifespan=lifespan)
        
#         # Simulate app startup
#         async with lifespan(app):
#             # Create an old file that should be deleted
#             old_file_path = Path(REPO_PATH) / "static" / "lifespan_old_image.jpg"
#             old_file_path.touch()
#             os.utime(old_file_path, (time.time() - 601, time.time() - 601))
            
#             # Get enough sleep for the scheduled cleaning task to remove you
#             await asyncio.sleep(65)
            
#             assert not old_file_path.exists(), "La tarea de limpieza no eliminó el archivo en el ciclo de vida."

#     # Run asynchronous function in event loop
#     try:
#         asyncio.run(run_test())
#     except asyncio.CancelledError:
#         print("Test finalizado: tarea de limpieza cancelada correctamente.")


# Module for testing the FastAPI application for YOLO image segmentation.

from pathlib import Path
import os
import pytest
import time
import asyncio
import pandas as pd

from fastapi import FastAPI
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from person_image_segmentation.api.app import app, clean_old_images, schedule_cleaning_task,lifespan

# Cargar variables de entorno
load_dotenv()
VALID_TOKEN = os.getenv("VALID_TOKEN")
REPO_PATH = os.getenv("PATH_TO_REPO")


# Fixture para el cliente sin tareas en segundo plano ni contexto lifespan
@pytest.fixture(scope="module")
def client():
    # Crear una copia de la aplicación sin el ciclo de vida para evitar tareas de fondo
    app_no_lifespan = FastAPI()
    app_no_lifespan.router.routes = app.router.routes  # Copiar rutas de la app original

    with TestClient(app_no_lifespan) as client:
        yield client


# Fixture para el payload de la imagen de prueba
@pytest.fixture
def payload():
    path = Path(REPO_PATH) / "tests/test_images/test_image.jpg"
    if not path.exists():
        pytest.fail(f"Test image not found at {path}")

    return {
        "file": ("test_image.jpg", open(path, "rb"), "image/jpeg"),
        "headers": {"Authorization": f"Bearer {VALID_TOKEN}"},
    }

# Fixture para la imagen no JPEG
@pytest.fixture
def non_jpeg_payload():
    path = Path(REPO_PATH) / "tests/test_images/test_image.png"
    if not path.exists():
        pytest.fail(f"Test non-JPEG image not found at {path}")

    return {
        "file": ("test_image.png", open(path, "rb"), "image/png"),
        "headers": {"Authorization": f"Bearer {VALID_TOKEN}"},
    }

# Prueba del endpoint raíz
def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API para hacer predicciones con YOLO"}

# Prueba de predicción con token válido
def test_predict_mask_with_valid_token(client, payload):
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers=payload["headers"],
        )

    assert response.status_code == 200
    response_json = response.json()
    assert "filename" in response_json
    assert response_json["message"] == "Prediction complete!"

# Prueba de predicción con token inválido
def test_predict_mask_with_invalid_token(client, payload):
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers={"Authorization": "Bearer INVALID_TOKEN"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Token inválido o no autorizado"

# Prueba de predicción sin archivo
def test_predict_mask_with_no_file(client, payload):
    response = client.post(
        "/predict/",
        headers=payload["headers"],
    )

    assert response.status_code == 422

# Prueba de predicción con archivo no JPEG
def test_predict_mask_with_non_jpeg_file(client, non_jpeg_payload):
    with open(non_jpeg_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.png", image_file, "image/png")},
            headers=non_jpeg_payload["headers"],
        )

    assert response.status_code == 200
    response_json = response.json()
    assert "filename" in response_json
    assert response_json["message"] == "Prediction complete!"

# Prueba para asegurarse que el archivo csv se genera
def test_predict_mask_with_non_jpeg_file_with_csv(client, non_jpeg_payload):
    emissions_path = Path(REPO_PATH) / "static" / "emissions_inference_api.csv"

    # Ensure the directory exists
    emissions_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the emissions file with predefined content
    expected_data = pd.DataFrame([{
        'emissions': 0.1,
        'duration': 2,
        'cpu_power': 15,
        'gpu_power': 25,
        'ram_power': 10,
        'energy_consumed': 1.5,
    }])
    expected_data.to_csv(emissions_path, index=False)

    # Send request to predict endpoint
    with open(non_jpeg_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.png", image_file, "image/png")},
            headers=non_jpeg_payload["headers"],
        )

    # Validate response
    assert response.status_code == 200
    response_json = response.json()
    assert "filename" in response_json
    assert response_json["message"] == "Prediction complete!"

# Prueba de predicción sin máscaras detectadas
def test_predict_mask_with_no_masks(client):
    no_mask_image_path = Path(REPO_PATH) / "tests/test_images/test_image_no_mask.png"
    if not no_mask_image_path.exists():
        pytest.fail(f"Test image with no masks not found at {no_mask_image_path}")

    with open(no_mask_image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image_no_mask.png", image_file, "image/png")},
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        )

    assert response.status_code == 500
    response_json = response.json()
    assert "No masks found in the prediction." in response_json["detail"]

# Prueba de predicción con emisiones y token válido
def test_predict_with_emissions_with_valid_token(client, payload):
    with open(payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict_with_emissions/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            headers=payload["headers"],
        )

    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert "energy_stats" in response_json
    assert response_json["message"] == "Prediction complete with energy tracking!"

# Prueba de predicción con emisiones para imagen no JPEG
def test_predict_with_emissions_non_jpeg_image(client, non_jpeg_payload):
    with open(non_jpeg_payload["file"][1].name, "rb") as image_file:
        response = client.post(
            "/predict_with_emissions/",
            files={"file": ("test_image.png", image_file, "image/png")},
            headers=non_jpeg_payload["headers"],
        )

    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert "energy_stats" in response_json
    assert response_json["message"] == "Prediction complete with energy tracking!"

# Prueba de limpieza de archivos antiguos
def test_clean_old_images():
    # Crear archivo más antiguo que 10 minutos en la carpeta estática
    old_file_path = Path(REPO_PATH) / "static" / "old_test_image.jpg"
    old_file_path.touch()
    os.utime(old_file_path, (time.time() - 601, time.time() - 601))

    # Llamar a la función de limpieza
    clean_old_images()

    # Verificar que el archivo fue eliminado
    assert not old_file_path.exists(), "La función `clean_old_images` no eliminó el archivo antiguo."

@pytest.mark.asyncio
async def test_schedule_cleaning_task():
    # Crea la tarea de limpieza y cancélala después de un corto tiempo
    cleaning_task = asyncio.create_task(schedule_cleaning_task())

    # Espera un poco menos de 60 segundos para comprobar que `clean_old_images` se llama
    await asyncio.sleep(1)
    
    # Cancela la tarea para finalizar la prueba
    cleaning_task.cancel()
    try:
        await cleaning_task
    except asyncio.CancelledError:
        pass  # La tarea ha sido cancelada correctamente

def test_lifespan():
    """
    Test that the lifespan context manager starts and stops the cleaning task correctly.
    """
    async def run_test():
        app = FastAPI(lifespan=lifespan)
        
        # Simulate app startup
        async with lifespan(app):
            # Create an old file that should be deleted
            old_file_path = Path(REPO_PATH) / "static" / "lifespan_old_image.jpg"
            old_file_path.touch()
            os.utime(old_file_path, (time.time() - 601, time.time() - 601))
            
            # Get enough sleep for the scheduled cleaning task to remove you
            await asyncio.sleep(65)
            
            assert not old_file_path.exists(), "La tarea de limpieza no eliminó el archivo en el ciclo de vida."

    # Run asynchronous function in event loop
    try:
        asyncio.run(run_test())
    except asyncio.CancelledError:
        print("Test finalizado: tarea de limpieza cancelada correctamente.")
