"""Dobles de prueba puros (sin numpy/OpenCV) compartidos por los tests del microservicio."""

from __future__ import annotations

from pathlib import Path


class FrameFalso:
    """Imita lo mínimo de un array HxWxC: `.shape` y recorte `frame[y1:y2, x1:x2]`."""

    def __init__(self, ancho: int = 640, alto: int = 480, etiqueta: str = "f"):
        self.shape = (alto, ancho, 3)
        self.etiqueta = etiqueta

    def __getitem__(self, indice):
        ys, xs = indice
        alto = ys.stop - ys.start
        ancho = xs.stop - xs.start
        recorte = FrameFalso(ancho, alto, self.etiqueta)
        recorte.recorte = (xs.start, ys.start, xs.stop, ys.stop)
        return recorte

    def __repr__(self):
        return f"FrameFalso({self.etiqueta}, {self.shape})"


class WriterFalso:
    def __init__(self, ruta: Path, fps: float, ancho: int, alto: int):
        self.ruta = Path(ruta)
        self.fps = fps
        self.tamano = (ancho, alto)
        self.frames = []
        self.liberado = False

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.liberado = True
        # simula que el codec dejó un archivo en disco
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ruta.write_bytes(b"\x00" * max(len(self.frames), 1))


class FabricaWriters:
    """writer_factory inyectable que recuerda todos los writers creados."""

    def __init__(self):
        self.creados: list[WriterFalso] = []

    def __call__(self, ruta, fps, ancho, alto):
        w = WriterFalso(ruta, fps, ancho, alto)
        self.creados.append(w)
        return w


class ClienteS3Falso:
    def __init__(self, fallar: bool = False):
        self.subidas = []
        self.fallar = fallar

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        if self.fallar:
            raise RuntimeError("S3 caído (simulado)")
        self.subidas.append((filename, bucket, key, ExtraArgs))
