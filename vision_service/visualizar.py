"""
Modo visual de depuración: reproduce un video mostrando en vivo lo que el
pipeline va detectando -zonas de mesa y su estado, tracks de personas
(cliente vs. staff ya clasificado), y un aviso cuando se reporta una entrega-
superpuesto sobre el propio video.

No reporta nada a la API Django: usa un cliente en memoria (`ApiLocal`) con la
misma interfaz que `ApiClient`, así el motor de ocupación y la heurística de
staff funcionan exactamente igual que en producción, pero sin crear registros
de prueba en la base real.

Uso:
    python -m vision_service.visualizar "ruta/al/video.mp4"
    python -m vision_service.visualizar 0                     # cámara en vivo
    python -m vision_service.visualizar video.mp4 --velocidad 2   # 2x más rápido
    python -m vision_service.visualizar video.mp4 --config otra.json

Controles en la ventana: `q` o Esc para salir, cualquier otra tecla no hace nada.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from .pipeline import VisionPipeline

COLOR_LIBRE = (200, 200, 200)
COLOR_GRACIA = (0, 200, 255)
COLOR_OCUPADA = (0, 0, 255)
COLOR_STAFF = (255, 0, 255)
COLOR_CLIENTE = (0, 255, 0)


class ApiLocal:
    """Mismo contrato que ApiClient, pero en memoria (nada de HTTP/Django)."""

    def __init__(self):
        self._siguiente_ocupacion_id = 0
        self.ocupaciones: dict[int, dict] = {}
        self._siguiente_personal_id = 0
        self.personal_por_codigo: dict[str, dict] = {}
        self.entregas: list[dict] = []

    def abrir_ocupacion(self, mesa_id, inicio, num_personas, confianza):
        self._siguiente_ocupacion_id += 1
        oid = self._siguiente_ocupacion_id
        self.ocupaciones[oid] = {"mesa": mesa_id, "inicio": inicio, "fin": None}
        return {"id": oid}

    def cerrar_ocupacion(self, ocupacion_id, fin):
        self.ocupaciones[ocupacion_id]["fin"] = fin
        return {"id": ocupacion_id}

    def marcar_fusion(self, ocupacion_id, fusionada_con_id):
        return {}

    def adjuntar_clip_evidencia(self, ocupacion_id, clip_s3_key):
        return {"id": ocupacion_id}

    def obtener_o_crear_personal(self, codigo_tracking):
        if codigo_tracking not in self.personal_por_codigo:
            self._siguiente_personal_id += 1
            self.personal_por_codigo[codigo_tracking] = {
                "id": self._siguiente_personal_id,
                "codigo_tracking": codigo_tracking,
            }
        return self.personal_por_codigo[codigo_tracking]

    def reportar_entrega(self, ocupacion_id, tipo, timestamp, personal_id=None, confianza=0.0):
        entrega = {
            "ocupacion_id": ocupacion_id,
            "tipo": tipo,
            "timestamp": timestamp,
            "personal_id": personal_id,
        }
        self.entregas.append(entrega)
        return {"id": len(self.entregas)}


def _color_estado(estado: str):
    if estado == "ocupada":
        return COLOR_OCUPADA
    if estado in ("en_gracia_ocupando", "en_gracia_liberando"):
        return COLOR_GRACIA
    return COLOR_LIBRE


VENTANA = "Monitoreo de mesas (visualizacion, no escribe en la BD)"
SALTO_SEEK_SEGUNDOS = 5


def correr(ruta_video: str, ruta_config: str, velocidad: float = 1.0, fps_muestreo: float = 2.0):
    import cv2
    import numpy as np

    api = ApiLocal()
    pipeline = VisionPipeline(ruta_config=Path(ruta_config), api_client=api, evidencia=False)

    fuente = int(ruta_video) if str(ruta_video).isdigit() else ruta_video
    cap = cv2.VideoCapture(fuente)
    fps_original = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    salto_frames = max(int(fps_original / fps_muestreo), 1)
    espera_ms = max(int(1000 / fps_original / velocidad), 1)
    salto_seek_frames = int(fps_original * SALTO_SEEK_SEGUNDOS)

    inicio_wall = datetime.now()
    entregas_mostradas = 0
    banner_hasta = -1

    # último estado calculado (se actualiza solo en los frames muestreados, pero
    # se DIBUJA en todos los frames para que el overlay no parpadee)
    info_mesas: dict[int, str] = {mesa_id: "libre" for mesa_id in pipeline.config.mesas}
    tracks_dibujar: list[tuple] = []

    print("Controles: espacio = pausa/reanuda | d = +5s | a = -5s | q/Esc = salir")

    def procesar_y_dibujar(frame, frame_idx: int):
        nonlocal info_mesas, tracks_dibujar, banner_hasta, entregas_mostradas

        # reloj "de video" (no el reloj real del sistema): permite calcular el
        # tiempo de espera en CUALQUIER frame mostrado, no solo en los muestreados,
        # para que el contador avance suave en vez de saltar de a pasos.
        reloj_video = inicio_wall + timedelta(seconds=frame_idx / fps_original)

        if frame_idx % salto_frames == 0:
            pipeline.procesar_frame(frame, reloj_video)

            info_mesas = {
                mesa_id: pipeline.engine.estado_actual(mesa_id).value for mesa_id in pipeline.config.mesas
            }
            tracks_dibujar = [
                (t.bbox, t.track_id, pipeline.clasificador_staff.es_staff(t.track_id, reloj_video))
                for t in pipeline._ultimos_tracks
            ]

            if len(api.entregas) > entregas_mostradas:
                entregas_mostradas = len(api.entregas)
                banner_hasta = frame_idx + int(fps_original * 3)

        for mesa_id, mesa_cfg in pipeline.config.mesas.items():
            poligono = mesa_cfg["poligono"]
            estado = info_mesas.get(mesa_id, "libre")
            color = _color_estado(estado)
            pts = np.array(poligono, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 3)
            x, y = poligono[0]

            etiqueta = f"Mesa {mesa_id}: {estado}"
            inicio_ocupacion = pipeline.engine.inicio_ocupacion_actual(mesa_id)
            if inicio_ocupacion is not None:
                ocupacion_id = pipeline._ocupacion_api_id.get(mesa_id)
                entregas_ocupacion = [e for e in api.entregas if e["ocupacion_id"] == ocupacion_id]
                if entregas_ocupacion:
                    # ya hubo entrega: el contador se congela en ese momento, no
                    # sigue corriendo con el resto de la ocupación
                    ultima = max(e["timestamp"] for e in entregas_ocupacion)
                    espera_seg = max(int((ultima - inicio_ocupacion).total_seconds()), 0)
                    etiqueta += f" | entrega a los {espera_seg}s"
                else:
                    espera_seg = max(int((reloj_video - inicio_ocupacion).total_seconds()), 0)
                    etiqueta += f" | espera: {espera_seg}s"

            cv2.putText(
                frame, etiqueta, (int(x), max(int(y) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        for bbox, track_id, es_staff in tracks_dibujar:
            x1, y1, x2, y2 = map(int, bbox)
            color = COLOR_STAFF if es_staff else COLOR_CLIENTE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            etiqueta = f"staff #{track_id}" if es_staff else f"persona #{track_id}"
            cv2.putText(frame, etiqueta, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if frame_idx <= banner_hasta:
            cv2.putText(frame, "ENTREGA DETECTADA", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return frame

    def saltar_a(frame_idx: int):
        """Posiciona el capture en frame_idx (clamp a los límites del video) y devuelve el frame ya dibujado."""
        destino = max(frame_idx, 0)
        if total_frames is not None:
            destino = min(destino, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, destino)
        ok, frame = cap.read()
        if not ok:
            return None, frame_idx
        return procesar_y_dibujar(frame, destino), destino + 1

    frame_idx = 0
    frame_actual = None
    pausado = False

    try:
        while True:
            if not pausado:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_actual = procesar_y_dibujar(frame, frame_idx)
                frame_idx += 1

            if pausado:
                cv2.setWindowTitle(VENTANA, f"{VENTANA} [PAUSADO]")
            else:
                cv2.setWindowTitle(VENTANA, VENTANA)

            cv2.imshow(VENTANA, frame_actual)
            espera = espera_ms if not pausado else 30
            tecla = cv2.waitKey(espera) & 0xFF

            if tecla in (27, ord("q")):
                break
            elif tecla == ord(" "):
                pausado = not pausado
            elif tecla == ord("d"):
                nuevo_frame, frame_idx = saltar_a(frame_idx + salto_seek_frames)
                if nuevo_frame is not None:
                    frame_actual = nuevo_frame
            elif tecla == ord("a"):
                nuevo_frame, frame_idx = saltar_a(frame_idx - salto_seek_frames)
                if nuevo_frame is not None:
                    frame_actual = nuevo_frame
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Ocupaciones detectadas: {len(api.ocupaciones)}")
    print(f"Entregas detectadas: {len(api.entregas)}")
    print(f"Personal distinto identificado: {len(api.personal_por_codigo)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualización en vivo del pipeline (no escribe en la BD)")
    parser.add_argument("video", help="Ruta a la grabación (o índice de cámara en vivo, ej. 0)")
    parser.add_argument("--config", default="vision_service/config/mesas_zonas.json")
    parser.add_argument("--fps-muestreo", type=float, default=2.0)
    parser.add_argument("--velocidad", type=float, default=1.0, help="Multiplicador de velocidad de reproducción")
    args = parser.parse_args()

    correr(args.video, args.config, velocidad=args.velocidad, fps_muestreo=args.fps_muestreo)
