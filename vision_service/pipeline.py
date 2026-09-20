"""
Orquestador del pipeline de visión: lee un feed (cámara en vivo o grabación),
corre YOLOv8, actualiza el tracker, evalúa el overlap contra cada zona de
mesa, alimenta el OccupancyEngine y la heurística de personal, corre la
detección de fusión de mesas, graba el clip de evidencia de cada ocupación,
y reporta todo a la API Django.

Este archivo es el punto de entrada (`python -m vision_service.pipeline`) y es
el único módulo que requiere las dependencias pesadas (opencv-python,
ultralytics/torch, boto3); el resto del paquete (geometry, occupancy_engine,
staff_heuristic, table_fusion, clip_recorder, evidence_storage) es puro
Python y se puede testear sin ellas. Para tests, `VisionPipeline` acepta
detector/tracker/api_client/clip_recorder/storage inyectados.

Objetos abandonados sin persona: el pipeline distingue `hay_persona` (tracks
de personas solapando la zona) de `hay_objeto` (mochilas, platos, vasos) y se
lo pasa al motor, que aplica el timeout específico
`parametros.timeout_objeto_abandonado_segundos` cuando solo quedan objetos
(ver occupancy_engine.py).

Evidencia en S3: por cada mesa se mantiene un pre-roll de frames recortados a
la zona; al confirmarse la apertura se empieza a escribir un MP4 temporal y al
cerrarse la ocupación se sube a S3 y se adjunta la key al EventoOcupacion vía
PATCH (`clip_s3_key`). Si no hay bucket configurado (AWS_STORAGE_BUCKET_NAME
vacío) o se pasa `--sin-evidencia`, el pipeline funciona igual sin grabar.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from .api_client import ApiClient
from .clip_recorder import ClipRecorder
from .detector import DetectorYOLO
from .evidence_storage import S3EvidenceStorage
from .geometry import overlap_ratio
from .occupancy_engine import TIMEOUT_OBJETO_ABANDONADO_DEFAULT, OccupancyEngine
from .staff_heuristic import ClasificadorStaff
from .table_fusion import MesaOcupadaSnapshot, detectar_fusiones
from .tracker import TrackerPersonas

logger = logging.getLogger(__name__)

CLASES_PERSONA = {"person"}
# COCO (dataset de YOLOv8n) no tiene clase "plato": un plato liso sobre la mesa
# nunca va a matchear aquí. Se cubre lo más parecido disponible (cubiertos,
# vasos, comida) para que "objeto abandonado" dispare en más casos reales,
# pero sigue siendo un límite del modelo, no de esta lista.
CLASES_OBJETO_RELEVANTE = {
    "backpack", "handbag", "cup", "bottle", "bowl", "wine glass",
    "fork", "knife", "spoon",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
}


class ConfigMesas:
    def __init__(self, ruta_config: Path):
        data = json.loads(Path(ruta_config).read_text())
        self.camara_id = data["camara_id"]
        self.mesas = {m["mesa_id"]: m for m in data["mesas"]}
        self.parametros = data["parametros"]

    @property
    def periodo_gracia(self) -> timedelta:
        return timedelta(seconds=self.parametros["periodo_gracia_segundos"])

    @property
    def umbral_overlap_visita_staff(self) -> float:
        """
        Umbral de overlap para registrar que un track "visitó" una mesa (usado
        por la heurística de staff). Deliberadamente MÁS BAJO que
        umbral_overlap_ocupacion: un mesero pasando a dejar/recoger algo solo
        roza el borde de la zona (overlap parcial), mientras que confirmar que
        un cliente está sentado exige un overlap mucho mayor. Si se reutilizara
        el mismo umbral para ambas cosas, subir uno para filtrar falsos
        positivos de gente pasando cerca de la mesa termina también apagando
        la detección de visitas de staff.
        """
        return self.parametros.get("umbral_overlap_visita_staff", 0.15)

    @property
    def timeout_objeto_abandonado(self) -> timedelta:
        segundos = self.parametros.get("timeout_objeto_abandonado_segundos")
        if segundos is None:
            return TIMEOUT_OBJETO_ABANDONADO_DEFAULT
        return timedelta(seconds=segundos)

    @property
    def zonas(self) -> dict[int, list]:
        return {mesa_id: m["poligono"] for mesa_id, m in self.mesas.items()}


class VisionPipeline:
    def __init__(
        self,
        ruta_config: Path,
        api_client: ApiClient | None = None,
        detector=None,
        tracker=None,
        clip_recorder: ClipRecorder | None = None,
        storage: S3EvidenceStorage | None = None,
        evidencia: bool = True,
        fps_muestreo: float = 2.0,
    ):
        self.config = ConfigMesas(ruta_config)
        self.detector = detector or DetectorYOLO()
        self.tracker = tracker or TrackerPersonas()
        self.engine = OccupancyEngine(
            periodo_gracia=self.config.periodo_gracia,
            timeout_objeto_abandonado=self.config.timeout_objeto_abandonado,
        )
        self.clasificador_staff = ClasificadorStaff()
        self.api = api_client or ApiClient()
        # mesa_id -> id de EventoOcupacion abierto en la API
        self._ocupacion_api_id: dict[int, int] = {}
        # (track_id, mesa_id) -> inicio de la visita en curso (overlap sostenido)
        self._visita_activa: dict[tuple[int, int], datetime] = {}
        # expuestos para el modo visual (visualizar.py); no se usan en el flujo normal
        self._ultimos_tracks: list = []
        self._ultimos_objetos: list = []

        # --- evidencia (clips a S3) ---
        self.storage = storage or S3EvidenceStorage()
        self.conservar_clips_locales = os.environ.get("EVIDENCIA_CONSERVAR_LOCAL", "0") == "1"
        if not evidencia:
            self.clips = None
        elif clip_recorder is not None:
            self.clips = clip_recorder
        elif self.storage.habilitado or self.conservar_clips_locales:
            self.clips = ClipRecorder(
                zonas=self.config.zonas,
                dir_temp=self._dir_clips_locales(),
                fps=fps_muestreo,
                preroll_frames=int(self.config.periodo_gracia.total_seconds() * fps_muestreo),
            )
        else:
            logger.warning(
                "AWS_STORAGE_BUCKET_NAME vacío y EVIDENCIA_CONSERVAR_LOCAL!=1: "
                "no se grabarán clips de evidencia"
            )
            self.clips = None

    @staticmethod
    def _dir_clips_locales() -> Path:
        return Path(os.environ.get("EVIDENCIA_DIR_LOCAL") or Path(tempfile.gettempdir()) / "evidencia_clips")

    # --- procesamiento por frame -----------------------------------------------------

    def procesar_frame(self, frame, timestamp: datetime):
        detecciones = self.detector.detectar(frame)
        personas = [d for d in detecciones if d.clase in CLASES_PERSONA]
        objetos = [d for d in detecciones if d.clase in CLASES_OBJETO_RELEVANTE]

        tracks = self.tracker.actualizar(personas, frame)
        # expuestos para el modo visual (visualizar.py); no se usan en el flujo normal
        self._ultimos_tracks = tracks
        self._ultimos_objetos = objetos

        snapshots_ocupadas = []
        umbral_ocupacion = self.config.parametros["umbral_overlap_ocupacion"]
        umbral_liberacion = self.config.parametros["umbral_overlap_liberacion"]

        for mesa_id, mesa_cfg in self.config.mesas.items():
            overlaps_persona = [overlap_ratio(t.bbox, mesa_cfg["poligono"]) for t in tracks]
            overlaps_objeto = [overlap_ratio(o.bbox, mesa_cfg["poligono"]) for o in objetos]
            max_overlap_persona = max(overlaps_persona + [0.0])
            max_overlap_objeto = max(overlaps_objeto + [0.0])
            num_personas = sum(1 for r in overlaps_persona if r >= umbral_ocupacion)

            self._actualizar_visitas_mesa(
                mesa_id, tracks, overlaps_persona, self.config.umbral_overlap_visita_staff, timestamp
            )

            # histéresis simple: umbral más bajo para liberar que para ocupar,
            # así evitamos parpadeo en el borde de la zona.
            estado_previo = self.engine.estado_actual(mesa_id)
            if estado_previo.value in ("ocupada", "en_gracia_liberando"):
                umbral = umbral_liberacion
            else:
                umbral = umbral_ocupacion
            hay_persona = max_overlap_persona >= umbral
            hay_objeto = max_overlap_objeto >= umbral
            hay_presencia = hay_persona or hay_objeto

            confianza = max([t.bbox and 1.0 for t in tracks] or [0.0])  # placeholder simple
            evento = self.engine.procesar(
                mesa_id, timestamp, hay_presencia, num_personas, confianza, hay_persona=hay_persona
            )

            if self.clips is not None:
                self.clips.registrar_frame(mesa_id, frame, timestamp)

            if evento is not None:
                self._despachar_evento(mesa_id, evento)

            if self.engine.estado_actual(mesa_id).value == "ocupada":
                snapshots_ocupadas.append(
                    MesaOcupadaSnapshot(
                        mesa_id=mesa_id,
                        capacidad=mesa_cfg["capacidad"],
                        num_personas_detectadas=num_personas,
                        adyacentes=mesa_cfg["adyacentes"],
                    )
                )

        fusiones = detectar_fusiones(
            snapshots_ocupadas, self.config.parametros["exceso_personas_para_fusion"]
        )
        for mesa_a, mesa_b in fusiones:
            self._despachar_fusion(mesa_a, mesa_b)

        self._cerrar_visitas_de_tracks_ausentes(tracks, timestamp)

    # --- personal por comportamiento + entregas ----------------------------------------

    def _actualizar_visitas_mesa(self, mesa_id, tracks, overlaps, umbral, timestamp):
        """Abre/cierra la visita de cada track a esta mesa según el overlap del frame."""
        ids_con_overlap = set()
        for track, overlap in zip(tracks, overlaps):
            key = (track.track_id, mesa_id)
            if overlap >= umbral:
                ids_con_overlap.add(track.track_id)
                self._visita_activa.setdefault(key, timestamp)
            elif key in self._visita_activa:
                self._cerrar_visita(key, timestamp)
        # cierra visitas de tracks que siguen en escena pero ya no solapan esta mesa
        for key in list(self._visita_activa):
            track_id, mesa_id_activa = key
            if mesa_id_activa == mesa_id and track_id not in ids_con_overlap:
                self._cerrar_visita(key, timestamp)

    def _cerrar_visitas_de_tracks_ausentes(self, tracks, timestamp):
        """Cierra visitas de tracks que ya no aparecen en el frame (salieron de escena)."""
        ids_presentes = {t.track_id for t in tracks}
        for key in list(self._visita_activa):
            track_id, _ = key
            if track_id not in ids_presentes:
                self._cerrar_visita(key, timestamp)

    def _cerrar_visita(self, key, fin):
        inicio = self._visita_activa.pop(key, None)
        if inicio is None:
            return
        track_id, mesa_id = key
        self.clasificador_staff.registrar_visita(track_id, mesa_id, inicio, fin)
        self._evaluar_entrega(track_id, mesa_id, fin)

    def _evaluar_entrega(self, track_id, mesa_id, timestamp):
        """
        Si esta visita ya alcanza para clasificar al track como staff y la mesa
        sigue ocupada por un cliente, se reporta una entrega. Las visitas previas
        del mismo track (antes de acumular 3 mesas distintas) no generan entrega
        retroactiva: hasta ese punto el sistema no tenía forma de distinguirlo de
        un cliente que solo pasó cerca de la mesa.
        """
        if not self.clasificador_staff.es_staff(track_id, timestamp):
            return
        ocupacion_id = self._ocupacion_api_id.get(mesa_id)
        if ocupacion_id is None:
            return
        personal = self.api.obtener_o_crear_personal(str(track_id))
        self.api.reportar_entrega(
            ocupacion_id,
            tipo="otro",
            timestamp=timestamp,
            personal_id=personal.get("id"),
            confianza=1.0,
        )
        logger.info(
            "Entrega reportada: mesa %s, ocupacion %s, personal %s", mesa_id, ocupacion_id, personal.get("id")
        )

    # --- despacho a la API ------------------------------------------------------------

    def _despachar_evento(self, mesa_id: int, evento):
        if evento.tipo == "abrir_ocupacion":
            resultado = self.api.abrir_ocupacion(
                mesa_id, evento.timestamp, evento.num_personas_detectadas, evento.confianza
            )
            self._ocupacion_api_id[mesa_id] = resultado["id"]
            if self.clips is not None:
                try:
                    self.clips.iniciar(mesa_id, evento.timestamp)
                except Exception:  # la evidencia nunca debe tumbar el monitoreo
                    logger.exception("No se pudo iniciar el clip de la mesa %s", mesa_id)
        elif evento.tipo == "cerrar_ocupacion":
            ocupacion_id = self._ocupacion_api_id.pop(mesa_id, None)
            if ocupacion_id is not None:
                logger.info("Mesa %s liberada (%s)", mesa_id, evento.motivo)
                self.api.cerrar_ocupacion(ocupacion_id, evento.timestamp)
            self._guardar_evidencia(mesa_id, ocupacion_id)

    def _guardar_evidencia(self, mesa_id: int, ocupacion_id: int | None):
        """Cierra el clip de la mesa, lo sube a S3 y adjunta la key a la ocupación."""
        if self.clips is None:
            return
        if ocupacion_id is None:
            self.clips.descartar(mesa_id)
            return
        try:
            inicio = self.clips.inicio_grabacion(mesa_id)
            ruta = self.clips.finalizar(mesa_id)
        except Exception:
            logger.exception("No se pudo finalizar el clip de la mesa %s", mesa_id)
            return
        if ruta is None:
            return
        if not self.storage.habilitado:
            logger.info("Clip local conservado en %s (sin bucket S3)", ruta)
            return
        try:
            key = self.storage.key_para(self.config.camara_id, mesa_id, inicio or datetime.now())
            self.storage.subir_clip(ruta, key)
            self.api.adjuntar_clip_evidencia(ocupacion_id, key)
        except Exception:
            logger.exception("Falló la subida/registro del clip %s de la ocupación %s", ruta, ocupacion_id)
            return
        if not self.conservar_clips_locales:
            try:
                Path(ruta).unlink(missing_ok=True)
            except OSError:
                pass

    def _despachar_fusion(self, mesa_a: int, mesa_b: int):
        oc_a = self._ocupacion_api_id.get(mesa_a)
        oc_b = self._ocupacion_api_id.get(mesa_b)
        if oc_a and oc_b:
            self.api.marcar_fusion(oc_a, oc_b)

    def finalizar(self):
        """Al terminar el stream/grabación: cierra clips abiertos sin subirlos (ocupación sin cerrar)."""
        if self.clips is not None:
            rutas = self.clips.finalizar_todo()
            for mesa_id, ruta in rutas.items():
                logger.info("Ocupación de mesa %s seguía abierta al terminar; clip parcial en %s", mesa_id, ruta)

    # --- entrada de video -------------------------------------------------------------

    def correr_sobre_video(self, ruta_video: str, fps_muestreo: float = 2.0):
        """Punto de entrada para procesar una grabación (recording) o cámara en vivo."""
        import cv2  # import perezoso: opencv-python

        if self.clips is not None:
            self.clips.fps = fps_muestreo

        cap = cv2.VideoCapture(ruta_video)
        fps_original = cap.get(cv2.CAP_PROP_FPS) or 30
        salto_frames = max(int(fps_original / fps_muestreo), 1)

        frame_idx = 0
        inicio_wall = datetime.now()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % salto_frames == 0:
                    timestamp_video = inicio_wall + timedelta(seconds=frame_idx / fps_original)
                    self.procesar_frame(frame, timestamp_video)
                frame_idx += 1
        finally:
            cap.release()
            self.finalizar()


if __name__ == "__main__":
    import argparse

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Pipeline de visión para monitoreo de mesas")
    parser.add_argument("video", help="Ruta a la grabación (o índice de cámara en vivo, ej. 0)")
    parser.add_argument("--config", default="vision_service/config/mesas_zonas.json")
    parser.add_argument("--fps-muestreo", type=float, default=2.0)
    parser.add_argument(
        "--sin-evidencia",
        action="store_true",
        help="No grabar ni subir clips de evidencia a S3 (útil para pruebas locales).",
    )
    args = parser.parse_args()

    pipeline = VisionPipeline(
        ruta_config=Path(args.config),
        evidencia=not args.sin_evidencia,
        fps_muestreo=args.fps_muestreo,
    )
    fuente = int(args.video) if args.video.isdigit() else args.video
    pipeline.correr_sobre_video(fuente, fps_muestreo=args.fps_muestreo)
