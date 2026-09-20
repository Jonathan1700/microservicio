"""
Wrapper delgado sobre Ultralytics YOLOv8 para detección de personas y objetos
(bolsos, platos, vasos) por frame. Aislado en su propio módulo para poder
mockearlo en pruebas sin necesitar el modelo/pesos reales ni una GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Deteccion:
    clase: str  # "person", "backpack", "cup", "plate", etc. (según el dataset/modelo)
    confianza: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 en píxeles


class DetectorYOLO:
    """
    Uso esperado:

        detector = DetectorYOLO(modelo_path="yolov8n.pt")
        detecciones = detector.detectar(frame)

    `ultralytics` es una dependencia pesada (torch incluido) y no se instala
    en este entorno de reconstrucción del proyecto; el import se hace de forma
    perezosa para que el resto del microservicio (config, geometría, engine de
    ocupación, cliente API) se pueda importar y testear sin GPU/torch.
    """

    def __init__(self, modelo_path: str = "yolov8n.pt", umbral_confianza: float = 0.4):
        self.modelo_path = modelo_path
        self.umbral_confianza = umbral_confianza
        self._modelo = None

    def _cargar_modelo(self):
        if self._modelo is None:
            from ultralytics import YOLO  # import perezoso

            self._modelo = YOLO(self.modelo_path)
        return self._modelo

    def detectar(self, frame) -> list[Deteccion]:
        modelo = self._cargar_modelo()
        resultados = modelo.predict(frame, verbose=False)[0]
        detecciones = []
        for box in resultados.boxes:
            conf = float(box.conf[0])
            if conf < self.umbral_confianza:
                continue
            clase_id = int(box.cls[0])
            clase = resultados.names.get(clase_id, str(clase_id))
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detecciones.append(Deteccion(clase=clase, confianza=conf, bbox=(x1, y1, x2, y2)))
        return detecciones
