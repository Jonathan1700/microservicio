"""
Wrapper sobre un tracker multi-objeto (SORT o DeepSORT) para mantener un ID
persistente por persona entre frames. DeepSORT (con embeddings de apariencia)
es el elegido para producción porque permite re-identificar personal cuando
sale y vuelve a entrar en cuadro; SORT (solo IoU+Kalman) sirve como fallback
liviano para pruebas o hardware sin GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    clase: str


class IdentidadPersistente:
    """
    Memoria de re-identificación de largo plazo, por encima de DeepSort.

    DeepSort borra un track tras `max_age` frames sin verlo: si esa misma
    persona reaparece después (salió de cuadro y volvió, o quedó tapada más
    tiempo del esperado), DeepSort ya no tiene memoria de ella y le asigna un
    track_id nuevo desde cero. Esta clase guarda el último embedding de
    apariencia de cada identidad ya vista en el video y, cuando aparece un
    track_id de DeepSort que nunca vimos, lo compara contra esa memoria: si
    hay un buen match de apariencia, reutiliza el id persistente en vez de
    exponer uno nuevo.
    """

    def __init__(self, umbral_similitud: float = 0.65):
        self.umbral_similitud = umbral_similitud
        self._embeddings: dict[int, object] = {}  # id persistente -> último embedding
        self._mapa_deepsort_a_persistente: dict[int, int] = {}
        self._siguiente_id = 1

    def resolver(self, deepsort_track_id: int, embedding) -> int:
        if deepsort_track_id in self._mapa_deepsort_a_persistente:
            persistente = self._mapa_deepsort_a_persistente[deepsort_track_id]
            if embedding is not None:
                self._embeddings[persistente] = embedding
            return persistente

        persistente = self._buscar_por_apariencia(embedding) if embedding is not None else None
        if persistente is None:
            persistente = self._siguiente_id
            self._siguiente_id += 1

        self._mapa_deepsort_a_persistente[deepsort_track_id] = persistente
        if embedding is not None:
            self._embeddings[persistente] = embedding
        return persistente

    def _buscar_por_apariencia(self, embedding):
        import numpy as np

        if not self._embeddings:
            return None
        norma_actual = np.linalg.norm(embedding)
        mejor_id, mejor_similitud = None, -1.0
        for pid, emb in self._embeddings.items():
            similitud = float(np.dot(embedding, emb) / (norma_actual * np.linalg.norm(emb) + 1e-9))
            if similitud > mejor_similitud:
                mejor_id, mejor_similitud = pid, similitud
        if mejor_similitud >= self.umbral_similitud:
            return mejor_id
        return None


class TrackerPersonas:
    """
    Uso esperado (una instancia persistente por cámara, alimentada frame a frame):

        tracker = TrackerPersonas(backend="deepsort")
        tracks = tracker.actualizar(detecciones_personas, frame)

    Igual que en detector.py, el backend real (deep_sort_realtime o filterpy
    para SORT) se importa de forma perezosa para no forzar esas dependencias
    pesadas en este entorno de reconstrucción.
    """

    def __init__(
        self,
        backend: str = "deepsort",
        max_age: int = 30,
        n_init: int = 2,
        max_cosine_distance: float = 0.4,
        umbral_similitud_reid: float = 0.65,
    ):
        self.backend = backend
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self._impl = None
        self._identidad = IdentidadPersistente(umbral_similitud=umbral_similitud_reid)

    def _cargar_backend(self):
        if self._impl is not None:
            return self._impl
        if self.backend == "deepsort":
            from deep_sort_realtime.deepsort_tracker import DeepSort

            # Valores por defecto de la librería (max_cosine_distance=0.2, n_init=3)
            # son muy estrictos para muestreo a pocos fps: entre dos frames
            # analizados la misma persona puede haberse movido/girado lo
            # suficiente como para que el embedding de apariencia no calce, y
            # DeepSort le asigna un track_id nuevo (fragmentación) en vez de
            # reconocerla. Se afloja el umbral de apariencia y se confirma el
            # track más rápido (n_init=2).
            self._impl = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_cosine_distance=self.max_cosine_distance,
            )
        elif self.backend == "sort":
            from sort import Sort  # implementación clásica de SORT (Bewley et al.)

            self._impl = Sort()
        else:
            raise ValueError(f"Backend de tracking desconocido: {self.backend}")
        return self._impl

    def actualizar(self, detecciones, frame=None) -> list[Track]:
        impl = self._cargar_backend()
        if self.backend == "deepsort":
            entradas = [
                ([d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]], d.confianza, d.clase)
                for d in detecciones
            ]
            resultados = impl.update_tracks(entradas, frame=frame)
            tracks = []
            for t in resultados:
                if not t.is_confirmed():
                    continue
                # DeepSort sigue devolviendo un track ya confirmado varios frames
                # después de que la última detección real que lo respalda
                # desapareció (coasting con predicción de Kalman), hasta que se
                # cumple max_age. Sin este filtro se dibujan/cuentan "fantasmas":
                # un recuadro sin persona real adentro que puede seguir
                # solapando una mesa y generando visitas/ocupaciones falsas.
                if t.time_since_update > 0:
                    continue
                x1, y1, x2, y2 = t.to_ltrb()
                try:
                    embedding = t.get_feature()
                except (IndexError, AttributeError):
                    embedding = None
                track_id = self._identidad.resolver(t.track_id, embedding)
                tracks.append(Track(track_id=track_id, bbox=(x1, y1, x2, y2), clase="person"))
            return tracks
        else:  # sort
            import numpy as np

            dets_np = np.array(
                [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confianza] for d in detecciones]
            ) if detecciones else np.empty((0, 5))
            resultados = impl.update(dets_np)
            return [
                Track(track_id=int(r[4]), bbox=(r[0], r[1], r[2], r[3]), clase="person")
                for r in resultados
            ]
