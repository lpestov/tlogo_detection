# TBank Logo Detector API

Короткая инструкция по сборке и запуску REST API детекции логотипа Т-Банка (FastAPI + Ultralytics YOLO).

## Требования
- Docker установлен
- Веса модели по пути `Runs/tbank_logo_v12/weights/best.pt`

## Сборка
```bash
docker build --network=host -t tbank-logo-detector .
```

## Запуск
```bash
docker run --rm -p 8000:8000 tbank-logo-detector
```

## Тест
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

## Docs
Откройте `http://localhost:8000/docs`.
