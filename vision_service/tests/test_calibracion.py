import json
import tempfile
import unittest
from pathlib import Path

from vision_service.calibracion import (
    PARAMETROS_DEFAULT,
    MesaCalibrada,
    SesionCalibracion,
    calcular_adyacencias,
    cargar_config_base,
    construir_config,
    distancia_entre_bboxes,
    guardar_config,
)
from vision_service.pipeline import ConfigMesas

CUADRADO_A = [(40, 120), (220, 120), (220, 300), (40, 300)]
CUADRADO_B = [(230, 120), (410, 120), (410, 300), (230, 300)]  # a 10 px de A
CUADRADO_C = [(600, 120), (780, 120), (780, 300), (600, 300)]  # lejos


class GeometriaCalibracionTestCase(unittest.TestCase):
    def test_distancia_entre_bboxes(self):
        self.assertEqual(distancia_entre_bboxes(CUADRADO_A, CUADRADO_B), 10)
        self.assertEqual(distancia_entre_bboxes(CUADRADO_A, CUADRADO_A), 0)
        self.assertEqual(distancia_entre_bboxes(CUADRADO_B, CUADRADO_C), 190)

    def test_calcular_adyacencias_por_cercania(self):
        mesas = [
            MesaCalibrada(1, CUADRADO_A),
            MesaCalibrada(2, CUADRADO_B),
            MesaCalibrada(3, CUADRADO_C),
        ]
        self.assertEqual(calcular_adyacencias(mesas, umbral_px=60), {1: [2], 2: [1], 3: []})
        self.assertEqual(calcular_adyacencias(mesas, umbral_px=5), {1: [], 2: [], 3: []})


class SesionCalibracionTestCase(unittest.TestCase):
    def setUp(self):
        self.sesion = SesionCalibracion()

    def _marcar(self, puntos):
        for x, y in puntos:
            self.sesion.agregar_punto(x, y)

    def test_marcar_deshacer_y_cerrar(self):
        self._marcar(CUADRADO_A)
        self.assertEqual(len(self.sesion.poligono_actual), 4)
        self.sesion.deshacer_punto()
        self.assertEqual(len(self.sesion.poligono_actual), 3)
        mesa = self.sesion.cerrar_poligono(capacidad=6)
        self.assertEqual(mesa.mesa_id, 1)
        self.assertEqual(mesa.capacidad, 6)
        self.assertEqual(len(mesa.poligono), 3)
        self.assertEqual(self.sesion.poligono_actual, [])
        self.assertEqual(self.sesion.siguiente_mesa_id(), 2)

    def test_no_cierra_con_menos_de_tres_puntos(self):
        self._marcar(CUADRADO_A[:2])
        with self.assertRaises(ValueError):
            self.sesion.cerrar_poligono()
        self.assertIsNone(self.sesion.deshacer_punto() and None)  # deshacer no rompe

    def test_mesa_id_repetido_es_error(self):
        self._marcar(CUADRADO_A)
        self.sesion.cerrar_poligono(mesa_id=7)
        self._marcar(CUADRADO_B)
        with self.assertRaises(ValueError):
            self.sesion.cerrar_poligono(mesa_id=7)
        # el polígono en curso se conserva para poder corregir el id
        self.assertEqual(len(self.sesion.poligono_actual), 4)

    def test_eliminar_ultima_mesa(self):
        self.assertIsNone(self.sesion.eliminar_ultima_mesa())
        self._marcar(CUADRADO_A)
        self.sesion.cerrar_poligono()
        self.assertEqual(self.sesion.eliminar_ultima_mesa().mesa_id, 1)
        self.assertEqual(self.sesion.mesas, [])


class ConstruirConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_formato_compatible_con_el_pipeline(self):
        sesion = SesionCalibracion()
        for puntos, cap in ((CUADRADO_A, 4), (CUADRADO_B, 4), (CUADRADO_C, 6)):
            for x, y in puntos:
                sesion.agregar_punto(x, y)
            sesion.cerrar_poligono(capacidad=cap)

        config = sesion.construir_config("cam-1-salon", parametros={"periodo_gracia_segundos": 30})
        self.assertEqual(config["camara_id"], "cam-1-salon")
        self.assertEqual([m["mesa_id"] for m in config["mesas"]], [1, 2, 3])
        self.assertEqual(config["mesas"][0]["poligono"], [[40, 120], [220, 120], [220, 300], [40, 300]])
        self.assertEqual(config["mesas"][0]["adyacentes"], [2])
        self.assertEqual(config["mesas"][2]["adyacentes"], [])
        self.assertEqual(config["mesas"][2]["capacidad"], 6)
        # parámetros: los pasados pisan a los default, el resto se completa
        self.assertEqual(config["parametros"]["periodo_gracia_segundos"], 30)
        self.assertEqual(
            config["parametros"]["timeout_objeto_abandonado_segundos"],
            PARAMETROS_DEFAULT["timeout_objeto_abandonado_segundos"],
        )

        ruta = guardar_config(config, Path(self.tmp.name) / "sub" / "mesas_zonas.json")
        self.assertTrue(ruta.exists())
        cfg = ConfigMesas(ruta)  # lo que usa el pipeline real
        self.assertEqual(set(cfg.mesas), {1, 2, 3})
        self.assertEqual(cfg.periodo_gracia.total_seconds(), 30)
        self.assertEqual(cfg.mesas[1]["adyacentes"], [2])

    def test_adyacencias_manuales_pisan_las_automaticas(self):
        mesas = [MesaCalibrada(1, CUADRADO_A), MesaCalibrada(3, CUADRADO_C)]
        config = construir_config("cam", mesas, adyacencias_manuales={1: [3], 3: [1]})
        self.assertEqual(config["mesas"][0]["adyacentes"], [3])

    def test_ids_repetidos_es_error(self):
        with self.assertRaises(ValueError):
            construir_config("cam", [MesaCalibrada(1, CUADRADO_A), MesaCalibrada(1, CUADRADO_B)])

    def test_cargar_config_base_roundtrip(self):
        mesas = [MesaCalibrada(5, CUADRADO_A, capacidad=2), MesaCalibrada(6, CUADRADO_B)]
        config = construir_config("cam-x", mesas, parametros={"periodo_gracia_segundos": 20})
        ruta = guardar_config(config, Path(self.tmp.name) / "base.json")
        camara_id, mesas_cargadas, parametros = cargar_config_base(ruta)
        self.assertEqual(camara_id, "cam-x")
        self.assertEqual([(m.mesa_id, m.capacidad) for m in mesas_cargadas], [(5, 2), (6, 4)])
        self.assertEqual(mesas_cargadas[0].poligono, [(40.0, 120.0), (220.0, 120.0), (220.0, 300.0), (40.0, 300.0)])
        self.assertEqual(parametros["periodo_gracia_segundos"], 20)
        # una sesión que parte de la base sigue numerando después del máximo
        sesion = SesionCalibracion(mesas=mesas_cargadas)
        self.assertEqual(sesion.siguiente_mesa_id(), 7)

    def test_config_de_ejemplo_del_repo_es_cargable(self):
        ruta = Path(__file__).resolve().parent.parent / "config" / "mesas_zonas.json"
        camara_id, mesas, parametros = cargar_config_base(ruta)
        self.assertTrue(camara_id)
        self.assertGreaterEqual(len(mesas), 1)
        self.assertIn("timeout_objeto_abandonado_segundos", parametros)


if __name__ == "__main__":
    unittest.main()
