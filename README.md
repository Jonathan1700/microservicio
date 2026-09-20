# Microservicio de visión — monitoreo de mesas

Microservicio de visión por computadora (OpenCV + YOLOv8 + tracking) que detecta
ocupación de mesas, identifica personal por comportamiento y reporta entregas,
para el sistema de monitoreo de mesas en restaurante
([proyecto de API/base de datos aquí](https://github.com/Jonathan1700/Construccion)).

Este servicio **nunca toca una base de datos directamente**: solo procesa video
(cámara en vivo o grabación) y reporta lo que detecta a la API Django por
HTTP/JSON (`vision_service/api_client.py`). Así se puede correr en una máquina
distinta a la API (ej. un mini-PC junto a las cámaras, apuntando a la API en EC2).

## Estructura

```
vision_service/
├── calibracion.py       Herramienta para marcar las zonas de mesa sobre un frame real
├── geometry.py          Overlap geométrico bbox vs zona de mesa
├── occupancy_engine.py  Máquina de estados: gracia + timeout de objeto abandonado
├── staff_heuristic.py   Clasificación de personal por comportamiento
├── table_fusion.py      Fusión de mesas adyacentes por exceso de personas
├── clip_recorder.py     Grabación del clip de evidencia por ocupación (MP4)
├── evidence_storage.py  Subida del clip a S3 (boto3)
├── api_client.py        Cliente HTTP hacia la API Django
├── detector.py / tracker.py   Wrappers de YOLOv8 y DeepSORT (import perezoso)
├── pipeline.py          Orquestador end-to-end, punto de entrada CLI
├── visualizar.py        Modo visual de depuración (no escribe en la API real)
└── config/mesas_zonas.json    Calibración de zonas + parámetros
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # se carga automáticamente (python-dotenv)
```

`ultralytics` descarga automáticamente los pesos de YOLOv8n (`yolov8n.pt`) la
primera vez que se corre si no están presentes localmente.

## Calibración de zonas de mesa con video real

`vision_service/config/mesas_zonas.json` trae coordenadas de **ejemplo**. La
config real se genera marcando las mesas sobre un frame de la cámara del local:

```bash
python -m vision_service.calibracion grabaciones/salon-2026-09-10.mp4 \
    --camara-id cam-1-salon-principal \
    --salida vision_service/config/mesas_zonas.json
```

Controles en la ventana:

| Tecla / acción       | Efecto                                                                 |
|----------------------|--------------------------------------------------------------------------|
| click izquierdo      | agrega un vértice al polígono de la mesa en curso                       |
| click derecho o `z`  | deshace el último vértice                                               |
| `n` o Enter          | cierra el polígono; pide por terminal `mesa_id` y `capacidad`           |
| `d`                  | elimina la última mesa cerrada                                          |
| `f` / `b`            | avanza / retrocede 1 s en el video (para elegir un frame despejado)     |
| `s`                  | guarda el JSON (y `<salida>.preview.png`) y sale                        |
| `q` / Esc            | sale sin guardar                                                        |

Opciones útiles: `--frame N`, `--base config.json` (parte de una config
existente), `--umbral-adyacencia`, `--capacidad-default`, `--solo-ver`.

**`mesa_id` debe coincidir con el `id` de la `Mesa` en la API Django**
(`GET /api/mesas/`): crear primero las mesas ahí y usar esos ids al calibrar.

## Cómo correr el pipeline

```bash
export RESTAURANTE_API_URL=http://localhost:8000/api   # o en .env
python -m vision_service.pipeline ruta/a/grabacion.mp4
# o con cámara en vivo:
python -m vision_service.pipeline 0
# opciones: --config otra.json  --fps-muestreo 2  --sin-evidencia
```

## Modo visual de depuración

Para ver en vivo, sobre el propio video, las zonas de mesa (coloreadas por
estado), los tracks de personas (cliente vs. staff ya clasificado), el tiempo
de espera de cada mesa y un aviso cuando se detecta una entrega — **sin
escribir nada en la API real** (usa un cliente en memoria):

```bash
python -m vision_service.visualizar ruta/a/grabacion.mp4 --velocidad 2
```

Controles: `espacio` pausa/reanuda, `d`/`a` avanza/retrocede 5 s, `q`/Esc sale.

## Evidencia: clips a S3 por ocupación

Al cerrarse cada ocupación se sube a S3 un clip de esa ocupación (recortado a
la zona de la mesa) y se adjunta la key vía `PATCH /api/ocupaciones/<id>/`.
Configurar en `.env`: `AWS_STORAGE_BUCKET_NAME`, `AWS_S3_REGION_NAME`,
credenciales de AWS (o rol IAM si corre en EC2). Con el bucket vacío, el
pipeline sigue funcionando igual, simplemente no graba evidencia.

## Tests

```bash
python -m unittest discover -s vision_service/tests -t .
```

54+ tests puros (sin OpenCV/YOLO/boto3 reales, todo con dobles de prueba
inyectados); los de `test_opencv_smoke.py` se saltan si no hay OpenCV
instalado.

## Pendiente

- Asociación de entregas detectadas con el tipo real de plato/bebida (hoy
  siempre se reporta `tipo="otro"`: el modelo YOLOv8/COCO no distingue
  aperitivo/bebida/plato principal/cuenta, y COCO tampoco tiene clase "plato".
- Validar los umbrales de la heurística de staff (`staff_heuristic.py`) y de
  overlap (`config/mesas_zonas.json`) con más video real del local.
