"""Stdlib adversarial tests for disjoint D-2 synthetic morphology sweep."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from pareto_study import (CASE_NAMES, POLICIES, THRESHOLD, brush_up_route,
                          cell_sha, components, dominates, fixture, guidance_greedy,
                          png, run_bridge, study, weighted_link)


class ParetoTests(unittest.TestCase):
    def test_predeclared_distinct_structural_families(self):
        hashes = set()
        for name in CASE_NAMES:
            allowed, initial, scalar = fixture(name)
            self.assertGreater(len(components(initial)), 1)
            self.assertTrue(initial <= allowed)
            self.assertEqual(len(scalar), 32 * 32)
            self.assertTrue(all(0 <= value <= 65535 for value in scalar.values()))
            hashes.add((cell_sha(allowed), cell_sha(initial)))
        self.assertEqual(len(hashes), len(CASE_NAMES))

    def test_invalid_registered_family(self):
        with self.assertRaises(ValueError):
            fixture('unknown-seed')

    def test_cardinal_brush_up_respects_hard_wall(self):
        allowed = {(0, 0), (0, 1), (2, 0), (2, 1)}
        self.assertIsNone(brush_up_route((0, 0), (2, 0), allowed, 10))
        self.assertEqual(brush_up_route((0, 0), (0, 1), allowed, 1), [(0, 0), (0, 1)])

    def test_weight_and_guidance_never_override_hard_permissions(self):
        allowed, initial, scalar = fixture('sealed-halves')
        for weight in (0, 2, 8, 32):
            result, after = run_bridge(allowed, initial, scalar, 128, weight)
            self.assertGreater(result['components'], 1)
            self.assertEqual(result['terminal'], 'NO_BUDGETED_PERMITTED_LINK')
            self.assertTrue(after <= allowed)

    def test_refuse_to_convert_soft_guidance_to_permissions(self):
        allowed, initial, scalar = fixture('strait')
        permissive, _ = run_bridge(allowed, initial, scalar, 44, 0)
        strict, _ = run_bridge(allowed, initial, scalar, 44, 0, strict=True)
        self.assertLessEqual(permissive['components'], strict['components'])
        self.assertLessEqual(strict['below_soft_threshold_cells'], permissive['below_soft_threshold_cells'])

    def test_no_budget_link_preserves_initial_material(self):
        allowed, initial, scalar = fixture('strait')
        result, after = run_bridge(allowed, initial, scalar, 1)
        self.assertEqual(after, initial)
        self.assertEqual(result['new_cells'], 0)
        self.assertEqual(result['material_sha256'], cell_sha(initial))

    def test_painted_events_are_material_preserving_for_brush_up(self):
        allowed, initial, scalar = fixture('strait')
        result, after = run_bridge(allowed, initial, scalar, 22)
        current = set(initial)
        for event in result['events']:
            if event['kind'] in ('BRUSH_UP_ROUTE', 'BRIDGE_PATH_SYNTHETIC'):
                self.assertEqual(event['before'], event['after'])
                self.assertEqual(event['before'], cell_sha(current))
                self.assertIsNone(event['native_receipt'])
                self.assertTrue(all(p in allowed for p in map(tuple, event['path'])))
                self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1
                                    for a,b in zip(event['path'], event['path'][1:])))
            else:
                self.assertEqual(event['before'], cell_sha(current))
                p = tuple(event['point']); self.assertNotIn(p, current)
                current.add(p)
                self.assertEqual(event['after'], cell_sha(current))
        self.assertEqual(current, after)
        self.assertEqual(result['new_cells'], len(after-initial))

    def test_equal_actual_cells_and_greedy_control(self):
        for name in CASE_NAMES:
            allowed, initial, scalar = fixture(name)
            for weight in (0, 8):
                bridge, cells = run_bridge(allowed, initial, scalar, 44, weight)
                ctrl, ctrl_cells = guidance_greedy(allowed, initial, scalar, bridge['new_cells'])
                self.assertEqual(len(cells), len(ctrl_cells))
                self.assertEqual(bridge['new_cells'], ctrl['new_cells'])

    def test_weighted_determinism_and_no_invalid_reversal(self):
        allowed, initial, scalar = fixture('archipelago')
        self.assertEqual(weighted_link(initial, allowed, scalar, 30, 2),
                         weighted_link(initial, allowed, scalar, 30, 2))
        first, _ = run_bridge(allowed, initial, scalar, 44, 8)
        self.assertEqual(first, run_bridge(allowed, initial, scalar, 44, 8)[0])
        self.assertFalse(first['native_replay_verified'])
        self.assertEqual(first['native_receipts'], 0)

    def test_png_custody_and_decode(self):
        allowed, initial, _ = fixture('river-gate')
        raw = png(initial, allowed)
        self.assertEqual(raw[:8], b'\x89PNG\r\n\x1a\n')
        offset=8; chunks=[]
        while offset < len(raw):
            length = struct.unpack_from('>I', raw, offset)[0]
            kind = raw[offset+4:offset+8]
            body = raw[offset+8:offset+8+length]
            self.assertEqual(zlib.crc32(kind+body)&0xffffffff,
                             struct.unpack_from('>I', raw, offset+8+length)[0])
            chunks.append((kind,body)); offset += 12+length
        self.assertEqual([c[0] for c in chunks],[b'IHDR',b'IDAT',b'IEND'])
        self.assertEqual(len(zlib.decompress(chunks[1][1])),32*(1+32*4))

    def test_pareto_and_repeated_artifact_hashes(self):
        self.assertTrue(dominates({'components':1,'below_soft_threshold_cells':0,'new_cells':4},
                                  {'components':2,'below_soft_threshold_cells':1,'new_cells':5}))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            report1, h1 = study(a, 44)
            report2, h2 = study(b, 44)
            self.assertEqual(h1, h2)
            self.assertEqual(len(report1['cases']), len(CASE_NAMES))
            self.assertEqual(report1, report2)
            for name in CASE_NAMES:
                folder1, folder2 = Path(a)/name, Path(b)/name
                for path in folder1.iterdir():
                    self.assertEqual(path.read_bytes(), (folder2/path.name).read_bytes())
                self.assertEqual(len([*folder1.glob('*.png')]),1+2*len(POLICIES))
            self.assertEqual(hashlib.sha256((Path(a)/'pareto-study.json').read_bytes()).hexdigest(),h1)

    def test_detour_is_a_real_penalty_tradeoff(self):
        allowed, initial, scalar = fixture('guide-detour')
        near, _ = run_bridge(allowed, initial, scalar, 44, 0)
        guided, _ = run_bridge(allowed, initial, scalar, 44, 32)
        self.assertEqual(near['components'], 1)
        self.assertEqual(guided['components'], 1)
        self.assertLess(near['new_cells'], guided['new_cells'])
        self.assertGreater(near['below_soft_threshold_cells'], guided['below_soft_threshold_cells'])

    def test_invalid_budgets_rejected(self):
        with tempfile.TemporaryDirectory() as dest:
            for invalid in (-1,0,129):
                with self.assertRaises(ValueError):
                    study(dest,invalid)


if __name__ == '__main__':
    unittest.main()
