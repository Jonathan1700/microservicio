"""
Grabación de clips de evidencia por ocupación de mesa.

Por cada mesa se mantiene un pequeño buffer circular ("pre-roll") con los
últimos frames sampleados, recortados a la zona de la mesa (más un margen).
Cuando el OccupancyEngine confirma la apertura de una ocupación —cosa que
ocurre `periodo_gracia` segundos DESPUÉS de que la persona llegó— el pre-roll
permite que el clip arranque desde la llegada real y no desde la confirmación.
A partir de ahí cada frame sampleado se escribe de forma incremental a un
archivo temporal en disco (no se acumula en memoria: una ocupación puede durar
más de una hora). Al cerrarse la ocupación se finaliza el archivo y el
pipeline lo sube a S3 (ver evidence_storage.py).

Decisiones (documentadas también en el README):
  * Formato: MP4 (codec mp4v de OpenCV), un archivo por EventoOcupacion.
  * El clip es de la zona de la mesa recortada con margen (`margen_px`), no del
    frame completo: archivos mucho más chicos y la evidencia queda ligada a
    ESA mesa. El margen deja ver al personal acercándose.
  * FPS del clip = FPS de muestreo del pipeline (2 fps por defecto), así el
    clip reproduce el tiempo real de la ocupación.
  * `max_frames` corta la grabación (conserva el inicio) como tope de
    seguridad para ocupaciones anormalmente largas.

La escritura real usa OpenCV (`cv2.VideoWriter`) importado de forma perezosa;
en tests se inyecta un `writer_factory` falso, igual que en detector/tracker.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from .geometry import Point, polygon_to_bbox

logger = logging.getLogger(__name__)


class FrameWriter(Protocol):
    def write(self, frame) -> None: ...

    def release(self) -> None: ...


# (ruta_salida, fps, ancho, alto) -> FrameWriter
WriterFactory = Callable[[Path, float, int, int], FrameWriter]


def crear_writer_cv2(ruta: Path, fps: float, ancho: int, alto: int) -> FrameWriter:
    """Writer por defecto: cv2.VideoWriter MP4 (import perezoso de opencv-python)."""
    import cv2  # import perezoso

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(ruta), fourcc, float(fps), (int(ancho), int(alto)))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo abrir cv2.VideoWriter en {ruta}")
    return writer


def bbox_recorte(
    poligono: list[Point], ancho_frame: int, alto_frame: int, margen_px: int = 0
) -> tuple[int, int, int, int]:
    """Bounding box entero de la zona + margen, recortado a los límites del frame."""
    x1, y1, x2, y2 = polygon_to_bbox(poligono)
    x1 = max(0, int(x1) - margen_px)
    y1 = max(0, int(y1) - margen_px)
    x2 = min(int(ancho_frame), int(x2) + margen_px)
    y2 = min(int(alto_frame), int(y2) + margen_px)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"La zona {poligono} queda fuera del frame {ancho_frame}x{alto_frame}")
    return x1, y1, x2, y2


def recortar(frame, bbox: tuple[int, int, int, int]):
    """Recorta un frame (array HxWxC) a un bbox entero (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


@dataclass
class _GrabacionMesa:
    preroll: deque = field(default_factory=deque)
    writer: FrameWriter | None = None
    ruta: Path | None = None
    inicio: datetime | None = None
    frames_escritos: int = 0
    truncado: bool = False


class ClipRecorder:
    def __init__(
        self,
        zonas: dict[int, list[Point]],
        dir_temp: Path,
        fps: float = 2.0,
        preroll_frames: int = 90,
        max_frames: int = 20_000,
        margen_px: int = 40,
        writer_factory: WriterFactory = crear_writer_cv2,
    ):
        """
        zonas:          mesa_id -> polígono de la zona (misma config que el pipeline).
        dir_temp:       carpeta local donde se escriben los clips antes de subirlos.
        fps:            fps de escritura del clip (= fps de muestreo del pipeline).
        preroll_frames: frames previos a la apertura que se conservan por mesa
                        (a 2 fps, 90 frames = 45 s = período de gracia por defecto).
        max_frames:     tope de frames por clip (se conserva el inicio).
        margen_px:      margen alrededor de la zona de la mesa en el recorte.
        """
        self.zonas = zonas
        self.dir_temp = Path(dir_temp)
        self.fps = fps
        self.preroll_frames = preroll_frames
        self.max_frames = max_frames
        self.margen_px = margen_px
        self.writer_factory = writer_factory
        self._grabaciones: dict[int, _GrabacionMesa] = {}
        self._bboxes: dict[int, tuple[int, int, int, int]] = {}

    # --- helpers ------------------------------------------------------------------

    def _grabacion(self, mesa_id: int) -> _GrabacionMesa:
        g = self._grabaciones.get(mesa_id)
        if g is None:
            g = _GrabacionMesa(preroll=deque(maxlen=max(self.preroll_frames, 0)))
            self._grabaciones[mesa_id] = g
        return g

    def _bbox_de(self, mesa_id: int, frame) -> tuple[int, int, int, int]:
        bbox = self._bboxes.get(mesa_id)
        if bbox is None:
            alto, ancho = frame.shape[0], frame.shape[1]
            bbox = bbox_recorte(self.zonas[mesa_id], ancho, alto, self.margen_px)
            self._bboxes[mesa_id] = bbox
        return bbox

    def grabando(self, mesa_id: int) -> bool:
        return self._grabaciones.get(mesa_id, _GrabacionMesa()).writer is not None

    def inicio_grabacion(self, mesa_id: int) -> datetime | None:
        """Timestamp de inicio de la ocupación que se está grabando (None si no graba)."""
        g = self._grabaciones.get(mesa_id)
        return g.inicio if g is not None and g.writer is not None else None

    def ruta_local(self, mesa_id: int, inicio: datetime) -> Path:
        return self.dir_temp / f"mesa-{mesa_id}-{inicio:%Y%m%dT%H%M%S}.mp4"

    # --- API usada por el pipeline ---------------------------------------------------

    def registrar_frame(self, mesa_id: int, frame, timestamp: datetime | None = None) -> None:
        """Llamar por cada frame sampleado y cada mesa, esté o no ocupada."""
        if mesa_id not in self.zonas:
            return
        recorte = recortar(frame, self._bbox_de(mesa_id, frame))
        g = self._grabacion(mesa_id)
        if g.writer is None:
            if self.preroll_frames > 0:
                g.preroll.append(recorte)
            return
        self._escribir(g, recorte)

    def _escribir(self, g: _GrabacionMesa, recorte) -> None:
        if g.frames_escritos >= self.max_frames:
            if not g.truncado:
                g.truncado = True
                logger.warning("Clip %s alcanzó max_frames=%d; se conserva el inicio", g.ruta, self.max_frames)
            return
        g.writer.write(recorte)
        g.frames_escritos += 1

    def iniciar(self, mesa_id: int, inicio: datetime) -> Path:
        """Abrir el clip de una ocupación recién confirmada; vuelca el pre-roll."""
        if mesa_id not in self.zonas:
            raise KeyError(f"Mesa {mesa_id} no está en la calibración de zonas")
        g = self._grabacion(mesa_id)
        if g.writer is not None:
            self.descartar(mesa_id)
            g = self._grabacion(mesa_id)
        bbox = self._bboxes.get(mesa_id)
        if bbox is None:
            # todavía no pasó ningún frame por esta mesa: no hay de dónde sacar tamaño.
            raise RuntimeError(f"No hay frames registrados para la mesa {mesa_id} todavía")
        x1, y1, x2, y2 = bbox
        self.dir_temp.mkdir(parents=True, exist_ok=True)
        ruta = self.ruta_local(mesa_id, inicio)
        g.writer = self.writer_factory(ruta, self.fps, x2 - x1, y2 - y1)
        g.ruta = ruta
        g.inicio = inicio
        g.frames_escritos = 0
        g.truncado = False
        for recorte in g.preroll:
            self._escribir(g, recorte)
        g.preroll.clear()
        return ruta

    def finalizar(self, mesa_id: int) -> Path | None:
        """Cerrar el clip y devolver su ruta local (None si no se estaba grabando)."""
        g = self._grabaciones.get(mesa_id)
        if g is None or g.writer is None:
            return None
        g.writer.release()
        ruta, frames = g.ruta, g.frames_escritos
        self._grabaciones[mesa_id] = _GrabacionMesa(preroll=deque(maxlen=max(self.preroll_frames, 0)))
        if frames == 0:
            logger.warning("Clip %s sin frames; se descarta", ruta)
            try:
                Path(ruta).unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return ruta

    def descartar(self, mesa_id: int) -> None:
        """Cerrar y borrar el clip en curso sin subirlo (ej. la API rechazó la ocupación)."""
        ruta = self.finalizar(mesa_id)
        if ruta is not None:
            try:
                Path(ruta).unlink(missing_ok=True)
            except OSError:
                pass

    def finalizar_todo(self) -> dict[int, Path]:
        """Al terminar el video/stream: cierra los clips abiertos y devuelve sus rutas."""
        rutas = {}
        for mesa_id in list(self._grabaciones):
            ruta = self.finalizar(mesa_id)
            if ruta is not None:
                rutas[mesa_id] = ruta
        return rutas
