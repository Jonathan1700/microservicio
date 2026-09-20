import unittest

from vision_service.table_fusion import MesaOcupadaSnapshot, detectar_fusiones


class TableFusionTestCase(unittest.TestCase):
    def test_no_detecta_fusion_si_no_hay_exceso_de_personas(self):
        mesas = [
            MesaOcupadaSnapshot(mesa_id=1, capacidad=4, num_personas_detectadas=2, adyacentes=[2]),
            MesaOcupadaSnapshot(mesa_id=2, capacidad=4, num_personas_detectadas=2, adyacentes=[1]),
        ]
        self.assertEqual(detectar_fusiones(mesas), [])

    def test_detecta_fusion_por_exceso_de_personas(self):
        mesas = [
            MesaOcupadaSnapshot(mesa_id=1, capacidad=4, num_personas_detectadas=5, adyacentes=[2]),
            MesaOcupadaSnapshot(mesa_id=2, capacidad=4, num_personas_detectadas=3, adyacentes=[1]),
        ]
        # 5 + 3 = 8 personas > 4 + 4 - 1 = 7 -> fusión
        self.assertEqual(detectar_fusiones(mesas, exceso_personas_umbral=1), [(1, 2)])

    def test_no_fusiona_mesas_no_adyacentes(self):
        mesas = [
            MesaOcupadaSnapshot(mesa_id=1, capacidad=2, num_personas_detectadas=6, adyacentes=[]),
            MesaOcupadaSnapshot(mesa_id=3, capacidad=2, num_personas_detectadas=6, adyacentes=[]),
        ]
        self.assertEqual(detectar_fusiones(mesas), [])


if __name__ == "__main__":
    unittest.main()
