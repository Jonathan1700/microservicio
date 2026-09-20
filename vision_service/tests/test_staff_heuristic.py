import unittest
from datetime import datetime, timedelta

from vision_service.staff_heuristic import ClasificadorStaff


class ClasificadorStaffTestCase(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime(2026, 1, 1, 12, 0, 0)
        self.clasificador = ClasificadorStaff(
            ventana=timedelta(minutes=10),
            mesas_minimas_distintas=3,
            dwell_maximo_por_visita=timedelta(seconds=90),
        )

    def test_cliente_sentado_no_es_staff(self):
        track_cliente = 101
        self.clasificador.registrar_visita(
            track_cliente, mesa_id=1, inicio=self.t0, fin=self.t0 + timedelta(minutes=25)
        )
        self.assertFalse(self.clasificador.es_staff(track_cliente, ahora=self.t0 + timedelta(minutes=25)))

    def test_mesero_visitando_varias_mesas_es_staff(self):
        track_mesero = 202
        for i, mesa_id in enumerate([1, 2, 3]):
            inicio = self.t0 + timedelta(minutes=i * 2)
            self.clasificador.registrar_visita(track_mesero, mesa_id, inicio, inicio + timedelta(seconds=20))
        ahora = self.t0 + timedelta(minutes=6)
        self.assertTrue(self.clasificador.es_staff(track_mesero, ahora))
        self.assertEqual(self.clasificador.mesas_atendidas_recientes(track_mesero, ahora), {1, 2, 3})

    def test_visitas_fuera_de_ventana_no_cuentan(self):
        track = 303
        for i, mesa_id in enumerate([1, 2, 3]):
            inicio = self.t0 + timedelta(minutes=i * 2)
            self.clasificador.registrar_visita(track, mesa_id, inicio, inicio + timedelta(seconds=20))
        ahora_lejos = self.t0 + timedelta(hours=2)
        self.assertFalse(self.clasificador.es_staff(track, ahora_lejos))


if __name__ == "__main__":
    unittest.main()
