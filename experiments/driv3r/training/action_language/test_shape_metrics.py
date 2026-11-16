"""GAL-003 additive topology and path descriptors: synthetic geometry only."""
import unittest
from lab import HostRefusal, World
from shape_metrics import morphology, direction

class ShapeMetricsTests(unittest.TestCase):
    def test_line_has_two_ends_and_no_cycle(self):
        m=morphology({(1,1),(2,1),(3,1),(4,1)},[])
        self.assertEqual((m['components'],m['graph_cycle_rank'],m['endpoints'],m['junctions']),(1,0,2,0))
        self.assertEqual((m['span_x'],m['span_y'],m['longest_horizontal_run']),(4,1,4))

    def test_square_has_cycle_but_zero_ends(self):
        m=morphology({(1,1),(1,2),(2,1),(2,2)},[])
        self.assertEqual((m['graph_cycle_rank'],m['endpoints'],m['components']),(1,0,1))

    def test_disconnected_pixels_are_not_implicit_branches(self):
        m=morphology({(1,1),(10,10)},[])
        self.assertEqual((m['components'],m['isolated_pixels'],m['junctions']),(2,2,0))

    def test_empty_world_is_finite(self):
        m=morphology([],[])
        self.assertEqual(m['bbox'],None)
        self.assertEqual(m['bbox_fill_fraction'],0)
        self.assertEqual(m['longest_paint_run'],0)

    def test_native_excluded_synthetic_jog_bend(self):
        w=World(120000,'empty','jog')
        menu,token=w.offer()
        self.assertIn('JOG_E_S',menu)
        w.apply('JOG_E_S',token)
        m=morphology(w.own,w.trace)
        self.assertEqual((m['new_ink'],m['paint_path_bends'],m['longest_paint_run']),(2,1,1))
        self.assertEqual(len(w.protected & w.own),0)

    def test_penup_is_not_counted_as_paint_path(self):
        w=World(120000,'empty','lift')
        token=w.offer()[1]
        w.apply('LIFT_E_2',token)
        m=morphology(w.own,w.trace)
        self.assertEqual((m['new_ink'],m['paint_path_bends']),(0,0))

    def test_invalid_step_direction_fails_closed(self):
        with self.assertRaises(ValueError):direction((2,2),(4,2))
        with self.assertRaises(ValueError):morphology({(-1,1)},[])

if __name__=='__main__':unittest.main()
