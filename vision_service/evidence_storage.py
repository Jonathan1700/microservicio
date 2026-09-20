"""
Almacenamiento de clips de evidencia en S3.

Estructura de keys en el bucket (decisión de diseño, ver README):

    evidencia/<camara_id>/mesa-<mesa_id>/<YYYY>/<MM>/<DD>/ocupacion-<inicio ISO compacto>.mp4

Así los clips quedan navegables por cámara, mesa y fecha desde la consola de
S3 y la key es reproducible a partir de los datos del EventoOcupacion.

`boto3` se importa de forma perezosa (igual que OpenCV/YOLO en el resto del
paquete) para que la lógica sea testeable sin credenciales ni red: en tests se
inyecta un `cliente` falso con el mismo método `upload_file`.

Configuración por variables de entorno (ver .env.example):
    AWS_STORAGE_BUCKET_NAME   bucket destino; si está vacío la evidencia queda DESHABILITADA
    AWS_S3_REGION_NAME        región (default us-east-1)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   las lee boto3 por su cadena estándar
    EVIDENCIA_PREFIJO_S3      prefijo raíz en el bucket (default "evidencia")
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_NO_SEGURO = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(valor) -> str:
    """Deja solo caracteres seguros para una key de S3 / nombre de carpeta."""
    s = _NO_SEGURO.sub("-", str(valor)).strip("-")
    return s or "sin-id"


def generar_key(
    camara_id: str,
    mesa_id: int,
    inicio: datetime,
    prefijo: str = "evidencia",
    extension: str = "mp4",
) -> str:
    return (
        f"{_slug(prefijo)}/{_slug(camara_id)}/mesa-{_slug(mesa_id)}/"
        f"{inicio:%Y/%m/%d}/ocupacion-{inicio:%Y%m%dT%H%M%S}.{extension}"
    )


class S3EvidenceStorage:
    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        prefijo: str | None = None,
        cliente=None,
    ):
        self.bucket = bucket if bucket is not None else os.environ.get("AWS_STORAGE_BUCKET_NAME", "")
        self.region = region or os.environ.get("AWS_S3_REGION_NAME", "us-east-1")
        self.prefijo = prefijo or os.environ.get("EVIDENCIA_PREFIJO_S3", "evidencia")
        self._cliente = cliente

    @property
    def habilitado(self) -> bool:
        return bool(self.bucket)

    def _obtener_cliente(self):
        if self._cliente is None:
            import boto3  # import perezoso

            self._cliente = boto3.client("s3", region_name=self.region)
        return self._cliente

    def key_para(self, camara_id: str, mesa_id: int, inicio: datetime) -> str:
        return generar_key(camara_id, mesa_id, inicio, prefijo=self.prefijo)

    def subir_clip(self, ruta_local: Path, key: str) -> str:
        """Sube el archivo y devuelve la key. Lanza excepción si no hay bucket o falla la subida."""
        if not self.habilitado:
            raise RuntimeError("AWS_STORAGE_BUCKET_NAME no configurado: evidencia S3 deshabilitada")
        ruta_local = Path(ruta_local)
        self._obtener_cliente().upload_file(
            str(ruta_local), self.bucket, key, ExtraArgs={"ContentType": "video/mp4"}
        )
        logger.info("Clip subido a s3://%s/%s (%d bytes)", self.bucket, key, ruta_local.stat().st_size)
        return key

    def url_publica(self, key: str) -> str:
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
