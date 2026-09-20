"""
Cliente HTTP delgado para que el microservicio de visión reporte lo que
detecta a la API Django (monitoreo.urls). Mantiene el microservicio de visión
desacoplado del modelo de datos: solo habla HTTP/JSON.
"""

from __future__ import annotations

import os
from datetime import datetime

import requests


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 5.0):
        self.base_url = (base_url or os.environ.get("RESTAURANTE_API_URL", "http://localhost:8000/api")).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(f"{self.base_url}/{path.lstrip('/')}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _patch(self, path: str, payload: dict) -> dict:
        resp = requests.patch(f"{self.base_url}/{path.lstrip('/')}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # --- Ocupaciones -----------------------------------------------------------

    def abrir_ocupacion(
        self, mesa_id: int, inicio: datetime, num_personas: int, confianza: float
    ) -> dict:
        return self._post(
            "ocupaciones/",
            {
                "mesa": mesa_id,
                "inicio": inicio.isoformat(),
                "estado": "ocupada",
                "num_personas_detectadas": num_personas,
                "confianza_deteccion": confianza,
            },
        )

    def cerrar_ocupacion(self, ocupacion_id: int, fin: datetime) -> dict:
        return self._patch(f"ocupaciones/{ocupacion_id}/", {"fin": fin.isoformat(), "estado": "liberada"})

    def marcar_fusion(self, ocupacion_id: int, fusionada_con_id: int) -> dict:
        return self._patch(
            f"ocupaciones/{ocupacion_id}/",
            {"estado": "fusionada", "fusionada_con": fusionada_con_id},
        )

    def adjuntar_clip_evidencia(self, ocupacion_id: int, clip_s3_key: str) -> dict:
        """Guarda en la ocupación la key del clip de evidencia ya subido a S3."""
        return self._patch(f"ocupaciones/{ocupacion_id}/", {"clip_s3_key": clip_s3_key})

    # --- Personal ----------------------------------------------------------------

    def obtener_o_crear_personal(self, codigo_tracking: str) -> dict:
        resp = requests.get(
            f"{self.base_url}/personal/", params={"search": codigo_tracking}, timeout=self.timeout
        )
        resp.raise_for_status()
        resultados = resp.json().get("results", resp.json())
        for r in resultados:
            if r["codigo_tracking"] == codigo_tracking:
                return r
        return self._post("personal/", {"codigo_tracking": codigo_tracking})

    # --- Entregas ------------------------------------------------------------------

    def reportar_entrega(
        self,
        ocupacion_id: int,
        tipo: str,
        timestamp: datetime,
        personal_id: int | None = None,
        confianza: float = 0.0,
    ) -> dict:
        payload = {
            "evento_ocupacion": ocupacion_id,
            "tipo": tipo,
            "timestamp": timestamp.isoformat(),
            "confianza_deteccion": confianza,
        }
        if personal_id is not None:
            payload["personal"] = personal_id
        return self._post("entregas/", payload)
