import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from vision_service.evidence_storage import S3EvidenceStorage, generar_key
from vision_service.tests.fakes import ClienteS3Falso


class GenerarKeyTestCase(unittest.TestCase):
    def test_estructura_por_camara_mesa_y_fecha(self):
        key = generar_key("cam-1-salon-principal", 3, datetime(2026, 9, 11, 12, 30, 0))
        self.assertEqual(
            key, "evidencia/cam-1-salon-principal/mesa-3/2026/09/11/ocupacion-20260911T123000.mp4"
        )

    def test_sanea_caracteres_no_seguros(self):
        key = generar_key("cámara 1/salón", 2, datetime(2026, 1, 2, 3, 4, 5), prefijo="clips")
        self.assertEqual(key, "clips/c-mara-1-sal-n/mesa-2/2026/01/02/ocupacion-20260102T030405.mp4")


class S3EvidenceStorageTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.archivo = Path(self.tmp.name) / "clip.mp4"
        self.archivo.write_bytes(b"\x00" * 10)

    def tearDown(self):
        self.tmp.cleanup()

    def test_deshabilitado_sin_bucket(self):
        with mock.patch.dict(os.environ, {"AWS_STORAGE_BUCKET_NAME": ""}):
            storage = S3EvidenceStorage(cliente=ClienteS3Falso())
        self.assertFalse(storage.habilitado)
        with self.assertRaises(RuntimeError):
            storage.subir_clip(self.archivo, "evidencia/x.mp4")

    def test_configuracion_desde_entorno(self):
        env = {
            "AWS_STORAGE_BUCKET_NAME": "bucket-env",
            "AWS_S3_REGION_NAME": "sa-east-1",
            "EVIDENCIA_PREFIJO_S3": "pruebas",
        }
        with mock.patch.dict(os.environ, env):
            storage = S3EvidenceStorage(cliente=ClienteS3Falso())
        self.assertTrue(storage.habilitado)
        self.assertEqual(storage.bucket, "bucket-env")
        self.assertEqual(storage.region, "sa-east-1")
        key = storage.key_para("cam-1", 1, datetime(2026, 9, 11, 12, 0, 0))
        self.assertTrue(key.startswith("pruebas/cam-1/mesa-1/2026/09/11/"))
        self.assertEqual(storage.url_publica(key), f"https://bucket-env.s3.sa-east-1.amazonaws.com/{key}")

    def test_subir_clip_usa_upload_file_con_content_type(self):
        cliente = ClienteS3Falso()
        storage = S3EvidenceStorage(bucket="mi-bucket", region="us-east-1", cliente=cliente)
        key = storage.subir_clip(self.archivo, "evidencia/cam-1/mesa-1/2026/09/11/ocupacion-x.mp4")
        self.assertEqual(key, "evidencia/cam-1/mesa-1/2026/09/11/ocupacion-x.mp4")
        self.assertEqual(len(cliente.subidas), 1)
        filename, bucket, key_subida, extra = cliente.subidas[0]
        self.assertEqual(filename, str(self.archivo))
        self.assertEqual(bucket, "mi-bucket")
        self.assertEqual(key_subida, key)
        self.assertEqual(extra, {"ContentType": "video/mp4"})

    def test_error_de_s3_se_propaga(self):
        storage = S3EvidenceStorage(bucket="mi-bucket", cliente=ClienteS3Falso(fallar=True))
        with self.assertRaises(RuntimeError):
            storage.subir_clip(self.archivo, "evidencia/x.mp4")


if __name__ == "__main__":
    unittest.main()
