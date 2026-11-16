"""Freeze-only invariants; never evaluate policy outcomes in this module."""
import json
import struct
import unittest
from confirmation_fixtures import CASE_SEEDS, fixture, manifest, cells_sha, digest, encode

class FreezeTests(unittest.TestCase):
    def test_all_8_unique(self):
        self.assertEqual(len(CASE_SEEDS), 8)
        records=manifest()['cases']
        self.assertEqual(len({(c['hard_permitted_sha256'], c['initial_material_sha256']) for c in records}),8)
    def test_exact_manifest_reproduction_and_u16(self):
        a=manifest(); self.assertEqual(a,manifest())
        self.assertTrue(a['outcomes_unobserved_at_freeze'])
        for case in a['cases']:
            allowed, material, scalar=fixture(case['id'],case['seed'])
            self.assertLess(len(material),len(allowed)); self.assertTrue(material <= allowed)
            self.assertEqual(case['hard_permitted_sha256'],cells_sha(allowed))
            self.assertEqual(case['initial_material_sha256'],cells_sha(material))
            raw=struct.pack('<1024H',*(scalar[(x,y)] for y in range(32) for x in range(32)))
            self.assertEqual(len(raw),2048)
            self.assertEqual(case['guidance_u16le_sha256'],digest(raw))
    def test_no_unregistered_seed_or_case(self):
        for name, seed in [('new', 1401), ('hard-separator', 1409)]:
            with self.assertRaises(ValueError):fixture(name,seed)
    def test_no_native_or_model_field_in_registration(self):
        payload=encode(manifest()).decode()
        self.assertNotIn('native_receipt',payload)
        self.assertNotIn('model_output',payload)
    def test_replay_is_not_sensitive_to_ambient_random_state(self):
        import random
        random.seed(77)
        before=encode(manifest());[random.random() for _ in range(30)]
        self.assertEqual(before, encode(manifest()))
    def test_hard_separator_remains_unbridgeable_by_cardinal_moves(self):
        allowed,material,_=fixture('hard-separator',1408)
        self.assertTrue(all((15,y) not in allowed for y in range(32)))
        self.assertTrue(any(x<15 for x,y in material))
        self.assertTrue(any(x>15 for x,y in material))

if __name__=='__main__': unittest.main()
