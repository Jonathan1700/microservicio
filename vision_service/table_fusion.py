"""
Heurística de fusión de mesas: cuando dos (o más) mesas adyacentes están
ocupadas al mismo tiempo Y la cantidad de personas detectadas en conjunto
excede la suma de sus capacidades individuales, se asume que un grupo grande
juntó las mesas (común cuando el cliente elige mesa libremente, como en este
restaurante). Se reporta como fusión en vez de como dos ocupaciones separadas
para no inflar las métricas de uso por mesa.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MesaOcupadaSnapshot:
    mesa_id: int
    capacidad: int
    num_personas_detectadas: int
    adyacentes: list[int]


def detectar_fusiones(
    mesas_ocupadas: list[MesaOcupadaSnapshot],
    exceso_personas_umbral: int = 1,
) -> list[tuple[int, int]]:
    """
    Recibe el snapshot de todas las mesas actualmente ocupadas y devuelve pares
    (mesa_id_a, mesa_id_b) que se consideran fusionados en ese instante.

    Regla: A y B son adyacentes, ambas ocupadas, y
    (personas_A + personas_B) > (capacidad_A + capacidad_B) - exceso_personas_umbral
    es decir, el grupo está usando más sillas de las que cualquiera de las dos
    mesas por separado podría ofrecer con margen razonable.
    """
    por_id = {m.mesa_id: m for m in mesas_ocupadas}
    fusiones: list[tuple[int, int]] = []
    vistos: set[frozenset[int]] = set()

    for mesa in mesas_ocupadas:
        for adyacente_id in mesa.adyacentes:
            vecino = por_id.get(adyacente_id)
            if vecino is None:
                continue  # la mesa adyacente no está ocupada, no hay fusión
            par = frozenset({mesa.mesa_id, vecino.mesa_id})
            if par in vistos:
                continue
            vistos.add(par)

            personas_total = mesa.num_personas_detectadas + vecino.num_personas_detectadas
            capacidad_total = mesa.capacidad + vecino.capacidad
            if personas_total > capacidad_total - exceso_personas_umbral:
                fusiones.append((mesa.mesa_id, vecino.mesa_id))

    return fusiones
