"""
Integración del pipeline: clasificación de personal por comportamiento y
reporte de EventoEntrega cuando un track ya clasificado como staff visita una
mesa ocupada (sin OpenCV/YOLO reales).
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from vision_service.detector import Deteccion
from vision_service.pipeline import VisionPipeline
from vision_service.tests.fakes import FrameFalso
from vision_service.tracker import Track

CONFIG = {
    "camara_id": "cam-test",
    "mesas": [
        {"mesa_id": 1, "poligono": [[40, 120], [220, 120], [220, 300], [40, 300]], "capacidad": 4, "adyacentes": []},
        {"mesa_id": 2, "poligono": [[230, 120], [410, 120], [410, 300], [230, 300]], "capacidad": 4, "adyacentes": []},
        {"mesa_id": 3, "poligono": [[420, 120], [600, 120], [600, 300], [420, 300]], "capacidad": 4, "adyacentes": []},
    ],
    "parametros": {
        "periodo_gracia_segundos": 5,
        "timeout_objeto_abandonado_segundos": 60,
        "umbral_overlap_ocupacion": 0.35,
        "umbral_overlap_liberacion": 0.10,
        "exceso_personas_para_fusion": 1,
    },
}

CLIENTE_MESA_1 = Deteccion(clase="person", confianza=0.9, bbox=(80, 150, 160, 280))
MESERO_MESA_1 = Deteccion(clase="person", confianza=0.9, bbox=(90, 160, 170, 290))
MESERO_MESA_2 = Deteccion(clase="person", confianza=0.9, bbox=(270, 160, 350, 290))
MESERO_MESA_3 = Deteccion(clase="person", confianza=0.9, bbox=(460, 160, 540, 290))


class DetectorFalso:
    def __init__(self):
        self.detecciones_actuales = []

    def detectar(self, frame):
        return list(self.detecciones_actuales)


class TrackerFalso:
    """
    Asigna a cada detección el track_id indicado explícitamente en `id_por_bbox`
    (simula la re-identificación por apariencia de DeepSORT: el mismo mesero
    puede aparecer con bboxes distintos -uno por mesa- pero mismo track_id).
    """

    def __init__(self, id_por_bbox: dict[tuple, int]):
        self._id_por_bbox = id_por_bbox

    def actualizar(self, detecciones, frame=None):
        return [Track(track_id=self._id_por_bbox[d.bbox], bbox=d.bbox, clase="person") for d in detecciones]


class ApiFalsa:
    def __init__(self):
        self.ocupaciones = []
        self.personal = {}
        self.entregas = []
        self._siguiente_ocupacion_id = 100
        self._siguiente_personal_id = 1

    def abrir_ocupacion(self, mesa_id, inicio, num_personas, confianza):
        self._siguiente_ocupacion_id += 1
        oid = self._siguiente_ocupacion_id
        self.ocupaciones.append({"id": oid, "mesa": mesa_id, "inicio": inicio, "fin": None})
        return {"id": oid}

    def cerrar_ocupacion(self, ocupacion_id, fin):
        for o in self.ocupaciones:
            if o["id"] == ocupacion_id:
                o["fin"] = fin
        return {"id": ocupacion_id}

    def marcar_fusion(self, ocupacion_id, fusionada_con_id):
        return {}

    def adjuntar_clip_evidencia(self, ocupacion_id, clip_s3_key):
        return {"id": ocupacion_id}

    def obtener_o_crear_personal(self, codigo_tracking):
        if codigo_tracking not in self.personal:
            self.personal[codigo_tracking] = {
                "id": self._siguiente_personal_id,
                "codigo_tracking": codigo_tracking,
            }
            self._siguiente_personal_id += 1
        return self.personal[codigo_tracking]

    def reportar_entrega(self, ocupacion_id, tipo, timestamp, personal_id=None, confianza=0.0):
        self.entregas.append(
            {
                "ocupacion_id": ocupacion_id,
                "tipo": tipo,
                "timestamp": timestamp,
                "personal_id": personal_id,
            }
        )
        return {"id": len(self.entregas)}


class PipelineEntregasTestCase(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime(2026, 9, 19, 12, 0, 0)
        self.tmp = tempfile.TemporaryDirectory()
        ruta_config = Path(self.tmp.name) / "mesas_zonas.json"
        ruta_config.write_text(json.dumps(CONFIG))

        self.detector = DetectorFalso()
        self.api = ApiFalsa()
        tracker = TrackerFalso(
            id_por_bbox={
                CLIENTE_MESA_1.bbox: 1,
                MESERO_MESA_1.bbox: 2,
                MESERO_MESA_2.bbox: 2,
                MESERO_MESA_3.bbox: 2,
            }
        )
        self.pipeline = VisionPipeline(
            ruta_config=ruta_config,
            api_client=self.api,
            detector=self.detector,
            tracker=tracker,
            evidencia=False,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _avanzar(self, desde_s, hasta_s, detecciones, paso_s=1):
        self.detector.detecciones_actuales = detecciones
        for s in range(desde_s, hasta_s + 1, paso_s):
            self.pipeline.procesar_frame(FrameFalso(640, 480, f"f{s}"), self.t0 + timedelta(seconds=s))

    def test_mesero_confirmado_reporta_entrega_en_mesa_ocupada(self):
        # Cliente se sienta en mesa 1 y se queda todo el tiempo (gracia: 5s)
        self._avanzar(0, 60, [CLIENTE_MESA_1])
        self.assertEqual(len(self.api.ocupaciones), 1)

        # El mismo track de "mesero" pasa brevemente por mesa 2 y mesa 3 (dwell
        # bajo) mientras el cliente sigue sentado -> tras la 2da mesa aún no es
        # staff (mínimo 3 mesas distintas)
        self._avanzar(61, 63, [CLIENTE_MESA_1, MESERO_MESA_2])
        self._avanzar(70, 72, [CLIENTE_MESA_1, MESERO_MESA_3])
        self.assertEqual(self.api.entregas, [])

        # Ahora visita mesa 1 (ya ocupada por el cliente) brevemente: 3ra mesa
        # distinta -> se confirma como staff en esta misma visita y se reporta
        # la entrega contra la ocupación abierta de mesa 1.
        self._avanzar(80, 82, [CLIENTE_MESA_1, MESERO_MESA_1])
        # el mesero se retira de la mesa (cierra la visita y dispara la evaluación)
        self._avanzar(83, 84, [CLIENTE_MESA_1])

        self.assertEqual(len(self.api.entregas), 1)
        entrega = self.api.entregas[0]
        self.assertEqual(entrega["ocupacion_id"], self.api.ocupaciones[0]["id"])
        self.assertEqual(entrega["tipo"], "otro")
        self.assertIsNotNone(entrega["personal_id"])

    def test_cliente_que_no_visita_otras_mesas_nunca_genera_entrega(self):
        self._avanzar(0, 120, [CLIENTE_MESA_1])
        self.assertEqual(self.api.entregas, [])

    def test_sin_ocupacion_abierta_no_reporta_entrega(self):
        # El "mesero" visita 3 mesas distintas pero ninguna está ocupada por un cliente
        self._avanzar(0, 2, [MESERO_MESA_1])
        self._avanzar(10, 12, [MESERO_MESA_2])
        self._avanzar(20, 22, [MESERO_MESA_3])
        self.assertEqual(self.api.entregas, [])


if __name__ == "__main__":
    unittest.main()
