"""Host-authoritative causal stack invariants, not native PAINT receipts."""
import unittest
from .world import World, specs
from .history_motif_lab import history

def surviving_moves(w):
    delta = {(0,-1):'N',(1,0):'E',(0,1):'S',(-1,0):'W'}
    return [delta[(b[0]-a[0],b[1]-a[1])] for a,b in zip(w.path,w.path[1:])]

class LiveStackInvariant(unittest.TestCase):
    def test_backtrack_and_relocation_are_causal(self):
        w = World.create(specs('development',1)[0])
        for _ in range(6): w.apply('MOVE_E')
        self.assertEqual(surviving_moves(w),['E']*6)
        w.apply('BACKTRACK')
        self.assertEqual(surviving_moves(w),['E']*5)
        self.assertEqual(history(w)[0],6)
        self.assertEqual(len(w.ink),6)
        w.apply('RELOCATE_START')
        self.assertEqual(surviving_moves(w),[])
        self.assertEqual(len(w.ink),6)
        self.assertGreater(w.logical_moves,6)
