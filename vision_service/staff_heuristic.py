"""
Heurística de identificación de personal por comportamiento: como no hay
uniforme distintivo ni reconocimiento facial, un track (ID persistente del
tracker) se clasifica como "staff" cuando visita varias mesas distintas en una
ventana de tiempo corta, sin quedarse sentado en ninguna (dwell time bajo por
mesa). Un cliente, en cambio, permanece en una sola mesa por tiempos largos.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class VisitaMesa:
    mesa_id: int
    inicio: datetime
    fin: datetime

    @property
    def duracion(self) -> timedelta:
        return self.fin - self.inicio


@dataclass
class HistorialTrack:
    visitas: list[VisitaMesa] = field(default_factory=list)

    def registrar(self, mesa_id: int, inicio: datetime, fin: datetime):
        self.visitas.append(VisitaMesa(mesa_id, inicio, fin))


class ClasificadorStaff:
    """
    Parámetros por defecto pensados para el caso de uso real: el mesero pasa
    por una mesa unos segundos a dejar/recoger algo, no se sienta. Un cliente
    ocupa la mesa varios minutos.
    """

    def __init__(
        self,
        ventana: timedelta = timedelta(minutes=10),
        mesas_minimas_distintas: int = 3,
        dwell_maximo_por_visita: timedelta = timedelta(seconds=90),
    ):
        self.ventana = ventana
        self.mesas_minimas_distintas = mesas_minimas_distintas
        self.dwell_maximo_por_visita = dwell_maximo_por_visita
        self._historiales: dict[int, HistorialTrack] = defaultdict(HistorialTrack)

    def registrar_visita(self, track_id: int, mesa_id: int, inicio: datetime, fin: datetime):
        self._historiales[track_id].registrar(mesa_id, inicio, fin)

    def es_staff(self, track_id: int, ahora: datetime) -> bool:
        historial = self._historiales.get(track_id)
        if not historial:
            return False
        recientes = [v for v in historial.visitas if ahora - v.fin <= self.ventana]
        visitas_breves = [v for v in recientes if v.duracion <= self.dwell_maximo_por_visita]
        mesas_distintas = {v.mesa_id for v in visitas_breves}
        return len(mesas_distintas) >= self.mesas_minimas_distintas

    def mesas_atendidas_recientes(self, track_id: int, ahora: datetime) -> set[int]:
        historial = self._historiales.get(track_id)
        if not historial:
            return set()
        return {
            v.mesa_id
            for v in historial.visitas
            if ahora - v.fin <= self.ventana and v.duracion <= self.dwell_maximo_por_visita
        }
