"""
Utilidades de geometría para el análisis de solapamiento (overlap) entre las
cajas delimitadoras (bounding boxes) que devuelve YOLOv8 y las zonas fijas de
cada mesa, siguiendo el enfoque de "Geometric Overlap Analysis" del paper base
del proyecto (Risaldi et al., 2026).
"""

from __future__ import annotations

from typing import Iterable


Point = tuple[float, float]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def polygon_to_bbox(poligono: Iterable[Point]) -> BBox:
    xs = [p[0] for p in poligono]
    ys = [p[1] for p in poligono]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def overlap_ratio(deteccion_bbox: BBox, zona_poligono: list[Point]) -> float:
    """
    Fracción del bounding box de la detección que cae dentro de la zona de la
    mesa. Se usa la aproximación por bounding box de la zona (suficiente para
    zonas rectangulares/cuadriláteras simples como las de un salón de mesas);
    para polígonos irregulares se puede sustituir por un cálculo con Shapely.
    """
    zona_bbox = polygon_to_bbox(zona_poligono)
    inter = bbox_intersection_area(deteccion_bbox, zona_bbox)
    area_deteccion = bbox_area(deteccion_bbox)
    if area_deteccion == 0:
        return 0.0
    return inter / area_deteccion
