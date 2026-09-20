import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from vision_service.clip_recorder import ClipRecorder, bbox_recorte
from vision_service.tests.fakes import FabricaWriters, FrameFalso


class BboxRecorteTestCase(unittest.TestCase):
    def test_margen_y_recorte_a_limites_del_frame(self):
        poligono = [[10, 10], [100, 10], [100, 60], [10, 60]]
        self.assertEqual(bbox_recorte(poligono, 640, 480, margen_px=20), (0, 0, 120, 80))
        self.assertEqual(bbox_recorte(poligono, 110, 70, margen_px=20), (0, 0, 110, 70))

    def test_zona_fuera_del_frame_es_error(self):
        with self.assertRaises(ValueError):
            bbox_recorte([[700, 10], [800, 10], [800, 60], [700, 60]], 640, 480)


class ClipRecorderTestCase(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime(2026, 9, 11, 12, 0, 0)
        self.tmp = tempfile.TemporaryDirectory()
        self.fabrica = FabricaWriters()
        self.zonas = {
            1: [[40, 120], [220, 120], [220, 300], [40, 300]],
            2: [[230, 120], [410, 120], [410, 300], [230, 300]],
        }
        self.recorder = ClipRecorder(
            zonas=self.zonas,
            dir_temp=Path(self.tmp.name),
            fps=2.0,
            preroll_frames=3,
            max_frames=10,
            margen_px=10,
            writer_factory=self.fabrica,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _frame(self, i):
        return FrameFalso(640, 480, etiqueta=f"f{i}")

    def test_sin_ocupacion_no_crea_writer_ni_archivos(self):
        for i in range(5):
            self.recorder.registrar_frame(1, self._frame(i), self.t0 + timedelta(seconds=i))
        self.assertFalse(self.recorder.grabando(1))
        self.assertEqual(self.fabrica.creados, [])
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_iniciar_vuelca_preroll_y_graba_recortes(self):
        for i in range(5):  # preroll_frames=3 -> solo se conservan f2, f3, f4
            self.recorder.registrar_frame(1, self._frame(i), self.t0 + timedelta(seconds=i))
        ruta = self.recorder.iniciar(1, self.t0)
        self.assertTrue(self.recorder.grabando(1))
        self.assertEqual(self.recorder.inicio_grabacion(1), self.t0)
        self.assertEqual(ruta.name, "mesa-1-20260911T120000.mp4")

        w = self.fabrica.creados[0]
        self.assertEqual(w.fps, 2.0)
        # zona 40..220 x 120..300 + margen 10 -> 200x200
        self.assertEqual(w.tamano, (200, 200))
        self.assertEqual([f.etiqueta for f in w.frames], ["f2", "f3", "f4"])
        self.assertEqual(w.frames[0].recorte, (30, 110, 230, 310))

        self.recorder.registrar_frame(1, self._frame(5), self.t0 + timedelta(seconds=5))
        self.assertEqual([f.etiqueta for f in w.frames], ["f2", "f3", "f4", "f5"])

    def test_finalizar_libera_writer_y_devuelve_ruta(self):
        self.recorder.registrar_frame(1, self._frame(0), self.t0)
        self.recorder.iniciar(1, self.t0)
        self.recorder.registrar_frame(1, self._frame(1), self.t0)
        ruta = self.recorder.finalizar(1)
        self.assertTrue(self.fabrica.creados[0].liberado)
        self.assertTrue(ruta.exists())
        self.assertFalse(self.recorder.grabando(1))
        self.assertIsNone(self.recorder.finalizar(1))  # segunda vez: nada que cerrar

    def test_max_frames_conserva_el_inicio(self):
        self.recorder.registrar_frame(1, self._frame(0), self.t0)
        self.recorder.iniciar(1, self.t0)
        for i in range(1, 30):
            self.recorder.registrar_frame(1, self._frame(i), self.t0 + timedelta(seconds=i))
        w = self.fabrica.creados[0]
        self.assertEqual(len(w.frames), 10)
        self.assertEqual(w.frames[0].etiqueta, "f0")
        self.assertEqual(w.frames[-1].etiqueta, "f9")

    def test_mesas_independientes(self):
        self.recorder.registrar_frame(1, self._frame(0), self.t0)
        self.recorder.registrar_frame(2, self._frame(0), self.t0)
        self.recorder.iniciar(1, self.t0)
        self.recorder.registrar_frame(1, self._frame(1), self.t0)
        self.recorder.registrar_frame(2, self._frame(1), self.t0)
        self.assertTrue(self.recorder.grabando(1))
        self.assertFalse(self.recorder.grabando(2))
        self.assertEqual(len(self.fabrica.creados), 1)

    def test_descartar_borra_el_archivo(self):
        self.recorder.registrar_frame(1, self._frame(0), self.t0)
        ruta = self.recorder.iniciar(1, self.t0)
        self.recorder.descartar(1)
        self.assertFalse(ruta.exists())
        self.assertFalse(self.recorder.grabando(1))

    def test_iniciar_sin_frames_previos_es_error(self):
        with self.assertRaises(RuntimeError):
            self.recorder.iniciar(2, self.t0)

    def test_finalizar_todo(self):
        for mesa in (1, 2):
            self.recorder.registrar_frame(mesa, self._frame(0), self.t0)
            self.recorder.iniciar(mesa, self.t0)
        rutas = self.recorder.finalizar_todo()
        self.assertEqual(set(rutas), {1, 2})
        self.assertFalse(self.recorder.grabando(1))
        self.assertFalse(self.recorder.grabando(2))


if __name__ == "__main__":
    unittest.main()
