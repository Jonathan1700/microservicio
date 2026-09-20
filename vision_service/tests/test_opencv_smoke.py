"""
Smoke tests que SÍ usan OpenCV/numpy reales. Se saltan automáticamente si no
están instalados, así la suite sigue corriendo en entornos sin dependencias
pesadas (patrón del proyecto). Cubren los puntos de contacto reales con cv2:
el writer MP4 de los clips y el dibujado/lectura de frames de la calibración.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

try:
    import cv2  # noqa: F401
    import numpy as np

    HAY_OPENCV = True
except ImportError:  # pragma: no cover
    HAY_OPENCV = False


@unittest.skipUnless(HAY_OPENCV, "opencv-python/numpy no instalados")
class OpenCVSmokeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clip_recorder_escribe_mp4_real(self):
        from vision_service.clip_recorder import ClipRecorder

        zonas = {1: [[40, 120], [220, 120], [220, 300], [40, 300]]}
        recorder = ClipRecorder(zonas=zonas, dir_temp=self.dir, fps=2.0, preroll_frames=2, margen_px=10)
        t0 = datetime(2026, 9, 11, 12, 0, 0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            recorder.registrar_frame(1, frame, t0)
        ruta = recorder.iniciar(1, t0)
        for _ in range(4):
            recorder.registrar_frame(1, frame, t0)
        ruta_final = recorder.finalizar(1)
        self.assertEqual(ruta_final, ruta)
        self.assertTrue(ruta.exists())
        self.assertGreater(ruta.stat().st_size, 0)

        cap = cv2.VideoCapture(str(ruta))
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 200)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 200)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 6)  # 2 pre-roll + 4
        cap.release()

    def test_calibracion_dibuja_y_lee_imagen(self):
        from vision_service.calibracion import FuenteFrames, MesaCalibrada, SesionCalibracion, dibujar

        imagen = self.dir / "frame.png"
        cv2.imwrite(str(imagen), np.full((240, 320, 3), 90, dtype=np.uint8))
        fuente = FuenteFrames(str(imagen))
        frame = fuente.frame_en(0)
        self.assertEqual(frame.shape, (240, 320, 3))

        sesion = SesionCalibracion(mesas=[MesaCalibrada(1, [(10, 50), (100, 50), (100, 150), (10, 150)])])
        sesion.agregar_punto(200, 60)
        sesion.agregar_punto(300, 60)
        salida = dibujar(frame, sesion, "mensaje de prueba")
        self.assertEqual(salida.shape, frame.shape)
        self.assertFalse(np.array_equal(salida, frame))  # algo se dibujó
        fuente.liberar()

    def test_fuente_frames_salta_a_frame_de_video(self):
        from vision_service.calibracion import FuenteFrames

        video = self.dir / "v.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
        for i in range(10):  # cada frame con un gris distinto para reconocerlo
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
        writer.release()

        fuente = FuenteFrames(str(video))
        self.assertEqual(fuente.total_frames, 10)
        f0 = fuente.frame_en(0)
        f7 = fuente.frame_en(7)
        self.assertLess(int(f0.mean()), int(f7.mean()))
        self.assertEqual(fuente.frame_en(999).shape, (48, 64, 3))  # se recorta al último frame
        fuente.liberar()


if __name__ == "__main__":
    unittest.main()
