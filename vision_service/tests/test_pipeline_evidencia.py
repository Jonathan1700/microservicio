"""
Integración del pipeline (sin OpenCV/YOLO/boto3): persona vs objeto en el
motor de ocupación + clip de evidencia grabado, subido a S3 y adjuntado al
EventoOcupacion vía la API.
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from vision_service.clip_recorder import ClipRecorder
from vision_service.detector import Deteccion
from vision_service.evidence_storage import S3EvidenceStorage
from vision_service.pipeline import VisionPipeline
from vision_service.tests.fakes import ClienteS3Falso, FabricaWriters, FrameFalso
from vision_service.tracker import Track

CONFIG = {
    "camara_id": "cam-test",
    "mesas": [
        {"mesa_id": 1, "poligono": [[40, 120], [220, 120], [220, 300], [40, 300]], "capacidad": 4, "adyacentes": [2]},
        {"mesa_id": 2, "poligono": [[230, 120], [410, 120], [410, 300], [230, 300]], "capacidad": 4, "adyacentes": [1]},
    ],
    "parametros": {
        "periodo_gracia_segundos": 30,
        "timeout_objeto_abandonado_segundos": 60,
        "umbral_overlap_ocupacion": 0.35,
        "umbral_overlap_liberacion": 0.10,
        "exceso_personas_para_fusion": 1,
    },
}

PERSONA_MESA_1 = Deteccion(clase="person", confianza=0.9, bbox=(80, 150, 160, 280))
VASO_MESA_1 = Deteccion(clase="cup", confianza=0.8, bbox=(100, 200, 130, 240))
MOCHILA_MESA_2 = Deteccion(clase="backpack", confianza=0.8, bbox=(280, 180, 340, 260))


class DetectorFalso:
    def __init__(self):
        self.detecciones_actuales = []

    def detectar(self, frame):
        return list(self.detecciones_actuales)


class TrackerFalso:
    def actualizar(self, detecciones, frame=None):
        return [Track(track_id=i + 1, bbox=d.bbox, clase="person") for i, d in enumerate(detecciones)]


class ApiFalsa:
    def __init__(self):
        self.llamadas = []
        self._siguiente_id = 100

    def abrir_ocupacion(self, mesa_id, inicio, num_personas, confianza):
        self._siguiente_id += 1
        self.llamadas.append(("abrir", mesa_id, inicio, num_personas))
        return {"id": self._siguiente_id}

    def cerrar_ocupacion(self, ocupacion_id, fin):
        self.llamadas.append(("cerrar", ocupacion_id, fin))
        return {"id": ocupacion_id}

    def marcar_fusion(self, ocupacion_id, fusionada_con_id):
        self.llamadas.append(("fusion", ocupacion_id, fusionada_con_id))
        return {}

    def adjuntar_clip_evidencia(self, ocupacion_id, clip_s3_key):
        self.llamadas.append(("clip", ocupacion_id, clip_s3_key))
        return {"id": ocupacion_id, "clip_s3_key": clip_s3_key}

    def de_tipo(self, tipo):
        return [c for c in self.llamadas if c[0] == tipo]


class PipelineEvidenciaTestCase(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime(2026, 9, 11, 12, 0, 0)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir_tmp = Path(self.tmp.name)
        ruta_config = self.dir_tmp / "mesas_zonas.json"
        ruta_config.write_text(json.dumps(CONFIG))

        self.detector = DetectorFalso()
        self.api = ApiFalsa()
        self.s3 = ClienteS3Falso()
        self.fabrica = FabricaWriters()
        self.storage = S3EvidenceStorage(bucket="bucket-test", region="us-east-1", cliente=self.s3)
        self.recorder = ClipRecorder(
            zonas={m["mesa_id"]: m["poligono"] for m in CONFIG["mesas"]},
            dir_temp=self.dir_tmp / "clips",
            fps=2.0,
            preroll_frames=60,
            writer_factory=self.fabrica,
        )
        self.pipeline = self._crear_pipeline(ruta_config, self.storage, self.recorder)
        self.ruta_config = ruta_config

    def _crear_pipeline(self, ruta_config, storage, recorder):
        return VisionPipeline(
            ruta_config=ruta_config,
            api_client=self.api,
            detector=self.detector,
            tracker=TrackerFalso(),
            clip_recorder=recorder,
            storage=storage,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _avanzar(self, desde_s, hasta_s, detecciones, paso_s=1):
        """Alimenta un frame por segundo en [desde_s, hasta_s] con las detecciones dadas."""
        self.detector.detecciones_actuales = detecciones
        for s in range(desde_s, hasta_s + 1, paso_s):
            self.pipeline.procesar_frame(FrameFalso(640, 480, f"f{s}"), self.t0 + timedelta(seconds=s))

    def test_flujo_completo_persona_luego_objeto_abandonado(self):
        # 0..30 s: persona en mesa 1 -> a los 30 s se confirma la ocupación
        self._avanzar(0, 30, [PERSONA_MESA_1])
        self.assertEqual(self.api.de_tipo("abrir"), [("abrir", 1, self.t0, 1)])
        self.assertTrue(self.recorder.grabando(1))
        # el clip arrancó con el pre-roll (frames desde la llegada real, no desde la confirmación)
        writer = self.fabrica.creados[0]
        self.assertEqual(writer.frames[0].etiqueta, "f0")

        # 31..90 s: sigue comiendo
        self._avanzar(31, 90, [PERSONA_MESA_1, VASO_MESA_1])
        self.assertEqual(self.api.de_tipo("cerrar"), [])

        # 91.. : se va y deja el vaso. La gracia normal (30 s) NO libera...
        self._avanzar(91, 91 + 35, [VASO_MESA_1])
        self.assertEqual(self.api.de_tipo("cerrar"), [])
        # ...el timeout de objeto abandonado (60 s) sí
        self._avanzar(91 + 36, 91 + 60, [VASO_MESA_1])
        self.assertEqual(self.api.de_tipo("cerrar"), [("cerrar", 101, self.t0 + timedelta(seconds=91))])

        # clip: finalizado, subido a S3 con la key esperada y adjuntado a la ocupación 101
        self.assertTrue(writer.liberado)
        self.assertFalse(self.recorder.grabando(1))
        key_esperada = "evidencia/cam-test/mesa-1/2026/09/11/ocupacion-20260911T120000.mp4"
        self.assertEqual(len(self.s3.subidas), 1)
        self.assertEqual(self.s3.subidas[0][1:3], ("bucket-test", key_esperada))
        self.assertEqual(self.api.de_tipo("clip"), [("clip", 101, key_esperada)])
        # el temporal local se borró tras subir
        self.assertFalse(Path(self.s3.subidas[0][0]).exists())

    def test_objeto_solo_nunca_abre_ocupacion_ni_clip(self):
        self._avanzar(0, 200, [MOCHILA_MESA_2])
        self.assertEqual(self.api.llamadas, [])
        self.assertFalse(self.recorder.grabando(2))
        self.assertEqual(self.fabrica.creados, [])

    def test_sin_bucket_conserva_clip_local_y_no_llama_api(self):
        storage = S3EvidenceStorage(bucket="", cliente=self.s3)
        self.pipeline = self._crear_pipeline(self.ruta_config, storage, self.recorder)
        self._avanzar(0, 30, [PERSONA_MESA_1])
        self._avanzar(31, 31 + 30, [])
        self.assertEqual(len(self.api.de_tipo("cerrar")), 1)
        self.assertEqual(self.s3.subidas, [])
        self.assertEqual(self.api.de_tipo("clip"), [])
        self.assertTrue(self.fabrica.creados[0].ruta.exists())

    def test_fallo_de_s3_no_tumba_el_pipeline(self):
        storage = S3EvidenceStorage(bucket="bucket-test", cliente=ClienteS3Falso(fallar=True))
        self.pipeline = self._crear_pipeline(self.ruta_config, storage, self.recorder)
        self._avanzar(0, 30, [PERSONA_MESA_1])
        with self.assertLogs("vision_service.pipeline", level="ERROR") as logs:
            self._avanzar(31, 31 + 30, [])
        self.assertIn("Falló la subida", logs.output[0])
        self.assertEqual(len(self.api.de_tipo("cerrar")), 1)
        self.assertEqual(self.api.de_tipo("clip"), [])
        # sigue funcionando para la siguiente ocupación
        self._avanzar(70, 100, [PERSONA_MESA_1])
        self.assertEqual(len(self.api.de_tipo("abrir")), 2)

    def test_pipeline_sin_evidencia(self):
        pipeline = VisionPipeline(
            ruta_config=self.ruta_config,
            api_client=self.api,
            detector=self.detector,
            tracker=TrackerFalso(),
            storage=self.storage,
            evidencia=False,
        )
        self.assertIsNone(pipeline.clips)
        self.detector.detecciones_actuales = [PERSONA_MESA_1]
        for s in range(0, 31):
            pipeline.procesar_frame(FrameFalso(), self.t0 + timedelta(seconds=s))
        self.assertEqual(len(self.api.de_tipo("abrir")), 1)
        self.assertEqual(self.s3.subidas, [])


if __name__ == "__main__":
    unittest.main()
