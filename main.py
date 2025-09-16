from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
from ultralytics import YOLO
import io
import torch

app = FastAPI()


class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)


class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")


class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")


class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")


model = YOLO("weights/best.pt")
device = 0 if torch.cuda.is_available() else "cpu"
allowed_content_types = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail="Unsupported media type")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model.predict(image, device=device, verbose=False, conf=0.6)

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy
            if xyxy is None:
                continue
            boxes = xyxy.cpu().numpy().astype(int).tolist()
            for x_min, y_min, x_max, y_max in boxes:
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                        )
                    )
                )

        return DetectionResponse(detections=detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


