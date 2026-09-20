"""
Herramienta de calibración de zonas de mesa a partir de un frame real del local.

    python -m vision_service.calibracion ruta/a/grabacion.mp4 --camara-id cam-1-salon

Abre una ventana de OpenCV con un frame del video (o una imagen) y permite
marcar con clicks el polígono de cada mesa. Al guardar genera el JSON que
consume el pipeline (`vision_service/config/mesas_zonas.json`) con:

  * `poligono` de cada mesa en píxeles del frame,
  * `capacidad` (se pregunta por terminal al cerrar cada polígono),
  * `adyacentes` calculadas automáticamente por cercanía de zonas (umbral en
    píxeles, `--umbral-adyacencia`), editables a mano después,
  * `parametros` heredados de una config base (`--base`) o los valores por
    defecto del proyecto.

También escribe una imagen `<salida>.preview.png` con las zonas dibujadas,
útil para verificar la calibración y para el informe.

Controles en la ventana:
    click izquierdo   agrega un vértice al polígono en curso
    click derecho / z deshace el último vértice
    n  (o Enter)      cierra el polígono actual -> pide mesa_id y capacidad en la terminal
    d                 elimina la última mesa cerrada
    f / b             avanza / retrocede en el video (para elegir un frame despejado)
    s                 guarda el JSON y sale
    q / Esc           sale sin guardar

La lógica de sesión (`SesionCalibracion`, `calcular_adyacencias`,
`construir_config`) es puro Python y está testeada sin OpenCV; solo
`extraer_frame`, `dibujar` y `ejecutar_interactivo` importan cv2 de forma
perezosa, siguiendo el patrón del resto del paquete.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import Point, polygon_to_bbox
from .occupancy_engine import TIMEOUT_OBJETO_ABANDONADO_DEFAULT

PARAMETROS_DEFAULT = {
    "periodo_gracia_segundos": 45,
    "timeout_objeto_abandonado_segundos": int(TIMEOUT_OBJETO_ABANDONADO_DEFAULT.total_seconds()),
    "umbral_overlap_ocupacion": 0.35,
    "umbral_overlap_liberacion": 0.10,
    "exceso_personas_para_fusion": 1,
}

UMBRAL_ADYACENCIA_PX_DEFAULT = 60
CAPACIDAD_DEFAULT = 4


# --- lógica pura ---------------------------------------------------------------------


@dataclass
class MesaCalibrada:
    mesa_id: int
    poligono: list[Point]
    capacidad: int = CAPACIDAD_DEFAULT

    def a_dict(self, adyacentes: list[int]) -> dict:
        return {
            "mesa_id": self.mesa_id,
            "poligono": [[int(x), int(y)] for x, y in self.poligono],
            "capacidad": int(self.capacidad),
            "adyacentes": list(adyacentes),
        }


def distancia_entre_bboxes(a: list[Point], b: list[Point]) -> float:
    """Separación mínima (en px) entre los bounding boxes de dos polígonos; 0 si se tocan/solapan."""
    ax1, ay1, ax2, ay2 = polygon_to_bbox(a)
    bx1, by1, bx2, by2 = polygon_to_bbox(b)
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def calcular_adyacencias(
    mesas: list[MesaCalibrada], umbral_px: float = UMBRAL_ADYACENCIA_PX_DEFAULT
) -> dict[int, list[int]]:
    """Dos mesas son adyacentes si sus zonas están a menos de `umbral_px` píxeles."""
    ady: dict[int, list[int]] = {m.mesa_id: [] for m in mesas}
    for i, a in enumerate(mesas):
        for b in mesas[i + 1 :]:
            if distancia_entre_bboxes(a.poligono, b.poligono) <= umbral_px:
                ady[a.mesa_id].append(b.mesa_id)
                ady[b.mesa_id].append(a.mesa_id)
    return {k: sorted(v) for k, v in ady.items()}


def construir_config(
    camara_id: str,
    mesas: list[MesaCalibrada],
    parametros: dict | None = None,
    umbral_adyacencia_px: float = UMBRAL_ADYACENCIA_PX_DEFAULT,
    adyacencias_manuales: dict[int, list[int]] | None = None,
) -> dict:
    """Arma el dict con el formato exacto de config/mesas_zonas.json."""
    ids = [m.mesa_id for m in mesas]
    if len(ids) != len(set(ids)):
        raise ValueError(f"mesa_id repetido en la calibración: {ids}")
    adyacencias = calcular_adyacencias(mesas, umbral_adyacencia_px)
    if adyacencias_manuales:
        adyacencias.update({k: sorted(v) for k, v in adyacencias_manuales.items()})
    params = dict(PARAMETROS_DEFAULT)
    params.update(parametros or {})
    return {
        "_comentario": (
            "Generado con `python -m vision_service.calibracion`. Polígonos en píxeles del "
            "frame de la cámara indicada; `adyacentes` calculadas por cercanía "
            f"(<= {umbral_adyacencia_px}px), revisar a mano si hace falta. "
            "`mesa_id` debe coincidir con el id de la Mesa en la API Django."
        ),
        "camara_id": camara_id,
        "mesas": [m.a_dict(adyacencias.get(m.mesa_id, [])) for m in mesas],
        "parametros": params,
    }


def cargar_config_base(ruta: Path) -> tuple[str, list[MesaCalibrada], dict]:
    """Lee una config existente para reutilizar camara_id, mesas y parametros."""
    data = json.loads(Path(ruta).read_text())
    mesas = [
        MesaCalibrada(
            mesa_id=int(m["mesa_id"]),
            poligono=[(float(x), float(y)) for x, y in m["poligono"]],
            capacidad=int(m.get("capacidad", CAPACIDAD_DEFAULT)),
        )
        for m in data.get("mesas", [])
    ]
    return data.get("camara_id", ""), mesas, dict(data.get("parametros", {}))


@dataclass
class SesionCalibracion:
    """Estado de una sesión de marcado; independiente de la UI (recibe clicks/teclas)."""

    mesas: list[MesaCalibrada] = field(default_factory=list)
    poligono_actual: list[Point] = field(default_factory=list)

    def agregar_punto(self, x: float, y: float) -> None:
        self.poligono_actual.append((float(x), float(y)))

    def deshacer_punto(self) -> Point | None:
        return self.poligono_actual.pop() if self.poligono_actual else None

    def siguiente_mesa_id(self) -> int:
        return max((m.mesa_id for m in self.mesas), default=0) + 1

    def cerrar_poligono(self, mesa_id: int | None = None, capacidad: int = CAPACIDAD_DEFAULT) -> MesaCalibrada:
        if len(self.poligono_actual) < 3:
            raise ValueError("Un polígono de mesa necesita al menos 3 vértices")
        if mesa_id is None:
            mesa_id = self.siguiente_mesa_id()
        if any(m.mesa_id == mesa_id for m in self.mesas):
            raise ValueError(f"Ya existe una mesa con mesa_id={mesa_id}")
        if capacidad < 1:
            raise ValueError("La capacidad debe ser >= 1")
        mesa = MesaCalibrada(mesa_id=int(mesa_id), poligono=list(self.poligono_actual), capacidad=int(capacidad))
        self.mesas.append(mesa)
        self.poligono_actual = []
        return mesa

    def eliminar_ultima_mesa(self) -> MesaCalibrada | None:
        return self.mesas.pop() if self.mesas else None

    def construir_config(self, camara_id: str, parametros: dict | None = None, umbral_adyacencia_px: float = UMBRAL_ADYACENCIA_PX_DEFAULT) -> dict:
        return construir_config(camara_id, self.mesas, parametros, umbral_adyacencia_px)


def config_a_json(config: dict) -> str:
    """JSON legible: indentado, pero con cada vértice `[x, y]` y cada lista de ids en una sola línea."""
    texto = json.dumps(config, indent=2, ensure_ascii=False)
    # colapsa listas que solo contienen números (vértices, adyacentes) a una línea
    patron = re.compile(r"\[\s*((?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?)\s*\]", re.S)
    return patron.sub(lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", texto) + "\n"


def guardar_config(config: dict, ruta: Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(config_a_json(config), encoding="utf-8")
    return ruta


# --- parte con OpenCV (imports perezosos) ------------------------------------------------

_COLORES = [(0, 200, 0), (0, 140, 255), (255, 80, 80), (200, 0, 200), (0, 200, 200), (60, 60, 230)]


class FuenteFrames:
    """Video (archivo o índice de cámara) o imagen; permite saltar a un frame concreto."""

    def __init__(self, fuente: str | int):
        import cv2  # import perezoso

        self._cv2 = cv2
        self.fuente = fuente
        self.es_imagen = isinstance(fuente, str) and Path(fuente).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        if self.es_imagen:
            self._imagen = cv2.imread(str(fuente))
            if self._imagen is None:
                raise FileNotFoundError(f"No se pudo leer la imagen {fuente}")
            self.total_frames = 1
            self.fps = 1.0
        else:
            self._cap = cv2.VideoCapture(fuente)
            if not self._cap.isOpened():
                raise FileNotFoundError(f"No se pudo abrir el video/cámara {fuente}")
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def frame_en(self, indice: int):
        if self.es_imagen:
            return self._imagen.copy()
        if self.total_frames > 0:
            indice = max(0, min(indice, self.total_frames - 1))
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer el frame {indice} de {self.fuente}")
        return frame

    def liberar(self):
        if not self.es_imagen:
            self._cap.release()


def dibujar(frame, sesion: SesionCalibracion, mensaje: str = ""):
    """Devuelve una copia del frame con las mesas cerradas, el polígono en curso y la ayuda."""
    import cv2  # import perezoso
    import numpy as np

    salida = frame.copy()
    for i, mesa in enumerate(sesion.mesas):
        color = _COLORES[i % len(_COLORES)]
        pts = np.array([[int(x), int(y)] for x, y in mesa.poligono], dtype=np.int32)
        capa = salida.copy()
        cv2.fillPoly(capa, [pts], color)
        cv2.addWeighted(capa, 0.25, salida, 0.75, 0, salida)
        cv2.polylines(salida, [pts], isClosed=True, color=color, thickness=2)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        cv2.putText(salida, f"Mesa {mesa.mesa_id} (cap {mesa.capacidad})", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(salida, f"Mesa {mesa.mesa_id} (cap {mesa.capacidad})", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    actual = [(int(x), int(y)) for x, y in sesion.poligono_actual]
    for j, p in enumerate(actual):
        cv2.circle(salida, p, 4, (0, 255, 255), -1)
        if j > 0:
            cv2.line(salida, actual[j - 1], p, (0, 255, 255), 2)

    ayuda = "clic: punto | z/clic der: deshacer | n: cerrar mesa | d: borrar ultima | f/b: frame | s: guardar | q: salir"
    cv2.rectangle(salida, (0, 0), (salida.shape[1], 44 if mensaje else 24), (0, 0, 0), -1)
    cv2.putText(salida, ayuda, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if mensaje:
        cv2.putText(salida, mensaje, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return salida


def _preguntar_int(texto: str, default: int) -> int:
    while True:
        valor = input(f"{texto} [{default}]: ").strip()
        if not valor:
            return default
        try:
            return int(valor)
        except ValueError:
            print("  -> ingresá un número entero")


def ejecutar_interactivo(
    fuente: str | int,
    salida: Path,
    camara_id: str,
    indice_frame: int = 0,
    base: Path | None = None,
    capacidad_default: int = CAPACIDAD_DEFAULT,
    umbral_adyacencia_px: float = UMBRAL_ADYACENCIA_PX_DEFAULT,
    solo_ver: bool = False,
) -> Path | None:
    import cv2  # import perezoso

    parametros: dict = {}
    sesion = SesionCalibracion()
    if base is not None and Path(base).exists():
        camara_base, mesas_base, parametros = cargar_config_base(base)
        sesion.mesas = mesas_base
        camara_id = camara_id or camara_base
        print(f"Config base cargada: {len(mesas_base)} mesas, camara_id={camara_base!r}")
    camara_id = camara_id or "cam-1"

    frames = FuenteFrames(fuente)
    frame = frames.frame_en(indice_frame)
    salto = max(int(frames.fps), 1)  # 1 segundo de video por pulsación
    mensaje = f"frame {indice_frame}" + (" (solo ver)" if solo_ver else "")
    ventana = "Calibracion de zonas de mesa"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    def on_mouse(evento, x, y, flags, _param):
        nonlocal mensaje
        if solo_ver:
            return
        if evento == cv2.EVENT_LBUTTONDOWN:
            sesion.agregar_punto(x, y)
            mensaje = f"punto ({x},{y}) - {len(sesion.poligono_actual)} vertices"
        elif evento == cv2.EVENT_RBUTTONDOWN:
            sesion.deshacer_punto()
            mensaje = f"deshecho - {len(sesion.poligono_actual)} vertices"

    cv2.setMouseCallback(ventana, on_mouse)
    guardado: Path | None = None
    try:
        while True:
            cv2.imshow(ventana, dibujar(frame, sesion, mensaje))
            tecla = cv2.waitKey(30) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("s"):
                if solo_ver:
                    break
                config = sesion.construir_config(camara_id, parametros, umbral_adyacencia_px)
                guardado = guardar_config(config, salida)
                preview = Path(salida).with_suffix(".preview.png")
                cv2.imwrite(str(preview), dibujar(frame, sesion))
                print(f"Guardado {guardado} ({len(sesion.mesas)} mesas) y {preview}")
                break
            if solo_ver:
                if tecla == ord("f"):
                    indice_frame += salto
                    frame = frames.frame_en(indice_frame)
                    mensaje = f"frame {indice_frame}"
                elif tecla == ord("b"):
                    indice_frame = max(0, indice_frame - salto)
                    frame = frames.frame_en(indice_frame)
                    mensaje = f"frame {indice_frame}"
                continue
            if tecla == ord("z"):
                sesion.deshacer_punto()
                mensaje = f"deshecho - {len(sesion.poligono_actual)} vertices"
            elif tecla in (ord("n"), 13, 10):
                try:
                    if len(sesion.poligono_actual) < 3:
                        raise ValueError("Marcá al menos 3 vértices antes de cerrar la mesa")
                    mesa_id = _preguntar_int("mesa_id (debe coincidir con el id de la Mesa en la API)", sesion.siguiente_mesa_id())
                    capacidad = _preguntar_int("capacidad (sillas)", capacidad_default)
                    mesa = sesion.cerrar_poligono(mesa_id, capacidad)
                    mensaje = f"Mesa {mesa.mesa_id} cerrada ({len(mesa.poligono)} vertices)"
                except ValueError as e:
                    mensaje = str(e)
                    print(mensaje)
            elif tecla == ord("d"):
                mesa = sesion.eliminar_ultima_mesa()
                mensaje = f"Mesa {mesa.mesa_id} eliminada" if mesa else "no hay mesas que borrar"
            elif tecla == ord("f"):
                indice_frame += salto
                frame = frames.frame_en(indice_frame)
                mensaje = f"frame {indice_frame}"
            elif tecla == ord("b"):
                indice_frame = max(0, indice_frame - salto)
                frame = frames.frame_en(indice_frame)
                mensaje = f"frame {indice_frame}"
    finally:
        frames.liberar()
        cv2.destroyAllWindows()
    return guardado


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Calibración de zonas de mesa sobre un frame real del local")
    parser.add_argument("fuente", help="Ruta a video/grabación, imagen (png/jpg) o índice de cámara (ej. 0)")
    parser.add_argument("--salida", default="vision_service/config/mesas_zonas.json", help="JSON de salida")
    parser.add_argument("--camara-id", default="", help="Identificador de la cámara (ej. cam-1-salon-principal)")
    parser.add_argument("--frame", type=int, default=0, help="Índice del frame inicial del video")
    parser.add_argument("--base", default=None, help="Config existente para partir de sus mesas/parametros")
    parser.add_argument("--capacidad-default", type=int, default=CAPACIDAD_DEFAULT)
    parser.add_argument("--umbral-adyacencia", type=float, default=UMBRAL_ADYACENCIA_PX_DEFAULT,
                        help="Distancia máx. (px) entre zonas para marcarlas adyacentes")
    parser.add_argument("--solo-ver", action="store_true",
                        help="Solo dibujar la config de --base (o de --salida) sobre el frame para verificarla")
    args = parser.parse_args(argv)

    fuente: str | int = int(args.fuente) if args.fuente.isdigit() else args.fuente
    base = Path(args.base) if args.base else (Path(args.salida) if args.solo_ver else None)
    ejecutar_interactivo(
        fuente=fuente,
        salida=Path(args.salida),
        camara_id=args.camara_id,
        indice_frame=args.frame,
        base=base,
        capacidad_default=args.capacidad_default,
        umbral_adyacencia_px=args.umbral_adyacencia,
        solo_ver=args.solo_ver,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
