"""D-2 topology bridge DEVELOPMENT tests: only Python stdlib, no native/model."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from recovery_study import cell_digest, components, run_arm, read_episode
from topology_study import (bridge_prefix, cheapest_link, compare_case, groups,
                            material_from_episode, png_material, study)

CORPUS = Path(__file__).parent / "qualified-corpus"


def decode_simple_png(blob):
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    at = 8
    chunks = {}
    while at < len(blob):
        n = struct.unpack_from(">I", blob, at)[0]
        tag = blob[at+4:at+8]
        data = blob[at+8:at+8+n]
        assert zlib.crc32(tag + data) & 0xffffffff == struct.unpack_from(">I", blob, at+8+n)[0]
        chunks[tag] = data
        at += n+12
    w,h,depth,mode,*other = struct.unpack(">IIBBBBB",chunks[b"IHDR"])
    assert (depth,mode,*other) == (8,6,0,0,0)
    raw = zlib.decompress(chunks[b"IDAT"])
    assert len(raw) == h*(1+w*4)
    cells = set()
    for y in range(h):
        row = raw[y*(1+w*4):(y+1)*(1+w*4)]
        assert row[0] == 0
        for x in range(w):
            if row[1+4*x:5+4*x] == bytes((232,222,199,255)):
                cells.add((x,y))
            else:
                assert row[1+4*x:5+4*x] == bytes((24,33,43,255))
    return (w,h),cells


class TopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global CORPUS
        cls._scratch = None
        if not CORPUS.exists():
            from runner import run
            cls._scratch = tempfile.TemporaryDirectory()
            CORPUS = Path(cls._scratch.name) / "qualified-corpus"
            run(Path(__file__).parent / "corpus/development-recipes.json", CORPUS)

    @classmethod
    def tearDownClass(cls):
        if cls._scratch is not None:
            cls._scratch.cleanup()

    def test_connectivity_and_stable_ties(self):
        cells={(0,0),(0,1),(3,0),(3,1)}
        permitted={(x,y) for x in range(5) for y in range(3)}
        self.assertEqual([len(g) for g in groups(cells)],[2,2])
        path=cheapest_link(cells,permitted,2)
        self.assertEqual(path,cheapest_link(cells,permitted,2))
        self.assertEqual(sum(p not in cells for p in path),2)
        self.assertEqual(len(components(cells|set(path))),1)
        self.assertIsNone(cheapest_link(cells,permitted,1))

    def test_no_paint_through_protected_wall(self):
        cells={(0,0),(0,1),(3,0),(3,1)}
        allowed={(x,y) for x in (0,1,3,4) for y in range(4)}
        self.assertIsNone(cheapest_link(cells,allowed,20))

    def test_bridge_refuses_insufficient_budget_without_mutation(self):
        folder=CORPUS/'moore-reentry'
        recipe,curve,scalar,_=read_episode(folder)
        prefix=run_arm(dict(recipe,steps=136),curve,scalar,'routed_future')
        original=material_from_episode(prefix)
        allowed={(x,y) for x in range(96) for y in range(96)}
        result,after=bridge_prefix(prefix,allowed,2)
        self.assertEqual(after,original)
        self.assertEqual(result['terminal'],'NO_BUDGETED_PERMITTED_BRIDGE')
        self.assertEqual(result['new_material_cells'],0)
        self.assertEqual(result['events'],[])

    def test_no_route_never_paints(self):
        recipe,curve,scalar,_=read_episode(CORPUS/'moore-reentry')
        prefix=run_arm(dict(recipe,steps=136),curve,scalar,'routed_future')
        before=material_from_episode(prefix)
        # Material sites remain legal but all nonmaterial movement sites are prohibited.
        result,after=bridge_prefix(prefix,set(before),24)
        self.assertEqual(result['terminal'],'NO_BUDGETED_PERMITTED_BRIDGE')
        self.assertEqual(before,after)
        self.assertFalse(any(e['kind']=='BRIDGE_STROKE_SYNTHETIC' for e in result['events']))

    def test_standalone_moore_matched_same_cells_and_component_gain(self):
        record,cellsets,_=compare_case(CORPUS/'moore-reentry')
        arms=record['arms']
        self.assertTrue(record['match_equal_actual_cell_count'])
        self.assertEqual(arms['reference_curve']['components'],4)
        self.assertEqual(arms['equal_cells_curve']['components'],3)
        self.assertEqual(arms['bridged']['components'],1)
        self.assertEqual(arms['bridged']['new_material_cells'],8)
        self.assertEqual(arms['bridged']['material_cells'],144)
        self.assertEqual(arms['equal_cells_curve']['material_cells'],144)
        self.assertEqual(arms['bridged']['material_sha256'],cell_digest(cellsets['bridged']))
        for event in arms['bridged']['events']:
            if event['kind']=='BRUSH_UP_ROUTE':
                route=event['route']
                self.assertFalse(event['native_route_verified'])
                self.assertIsNone(event['native_receipt'])
                self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(route,route[1:])))
                self.assertEqual(event['material_before_sha256'],event['material_after_sha256'])
        self.assertFalse(arms['bridged']['native_replay_verified'])
        self.assertEqual(record,compare_case(CORPUS/'moore-reentry')[0])

    def test_nonfragmented_arm_does_not_paint_unneeded_bridge(self):
        for name in ('hilbert-reentry','serpentine-reentry'):
            record,_,_=compare_case(CORPUS/name)
            self.assertEqual(record['arms']['bridged']['new_material_cells'],0)
            self.assertEqual(record['arms']['bridged']['components'],1)
            self.assertEqual(record['arms']['bridged']['terminal'],'ALREADY_CONNECTED')

    def test_guidance_strict_vs_override_exposes_tradeoff(self):
        folder=CORPUS/'moore-reentry'
        recipe,curve,scalar,_=read_episode(folder)
        record,cellsets,_=compare_case(folder)
        self.assertEqual(record['soft_guidance_crossings_count'],8)
        self.assertEqual(len(record['soft_guidance_crossings']),8)
        self.assertTrue(all(scalar[tuple(p)] < recipe['exit_below']
                            for p in record['soft_guidance_crossings']))
        strict=record['arms']['bridge_guidance_strict']
        self.assertEqual(strict['new_material_cells'],0)
        self.assertEqual(strict['components'],3)
        self.assertEqual(strict['terminal'],'NO_BUDGETED_PERMITTED_BRIDGE')
        self.assertEqual(cellsets['bridge_guidance_strict'],cellsets['prefix_curve'])
        self.assertEqual(record['arms']['bridged']['components'],1)

    def test_brush_up_hop_cap_prevents_optimistic_relocation(self):
        recipe,curve,scalar,_=read_episode(CORPUS/'moore-reentry')
        prefix=run_arm(dict(recipe,steps=136),curve,scalar,'routed_future')
        before=material_from_episode(prefix)
        allowed={(x,y) for x in range(96) for y in range(96)}
        limited,after=bridge_prefix(prefix,allowed,24,max_route_hops=0)
        self.assertEqual(limited['terminal'],'NO_BRUSH_UP_ROUTE')
        self.assertEqual(limited['new_material_cells'],0)
        self.assertEqual(after,before)
        self.assertEqual(limited['events'][-1]['kind'],'RECOVERY_REFUSED')

    def test_png_views_roundtrip_original_pixel_palette(self):
        cells={(0,0),(2,1),(4,3)}
        encoded=png_material(cells,5,4)
        self.assertEqual(decode_simple_png(encoded),((5,4),cells))
        self.assertEqual(encoded,png_material(cells,5,4))
        with self.assertRaises(AssertionError):
            png_material({(5,0)},5,4)

    def test_study_reproducible_with_hashed_views_and_original_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first,sha=study(CORPUS,root/'a')
            second,other=study(CORPUS,root/'b')
            self.assertEqual(first,second)
            self.assertEqual(sha,other)
            self.assertEqual(len(first['cases']),3)
            self.assertFalse(first['native_replay_verified'])
            self.assertTrue(first['no_native_undo_or_inference'])
            for case in first['cases']:
                for arm,meta in case['counterfactual_views'].items():
                    blob=(root/'a'/meta['path']).read_bytes()
                    import hashlib
                    self.assertEqual(hashlib.sha256(blob).hexdigest(),meta['sha256'])
                    (w,h),cells=decode_simple_png(blob)
                    self.assertEqual((w,h),(96,96))
                    self.assertEqual(cell_digest(cells),meta['cells_sha256'])

    def test_reject_material_event_hash_tamper(self):
        recipe,curve,scalar,_=read_episode(CORPUS/'hilbert-reentry')
        result=run_arm(dict(recipe,steps=40),curve,scalar,'routed_future')
        result['events'][0]['material_after_sha256']='0'*64
        with self.assertRaises(AssertionError):
            material_from_episode(result)

    def test_reject_reserve_outside_budget(self):
        for reserve in (0,160,161,-1):
            with self.assertRaises(ValueError):
                compare_case(CORPUS/'moore-reentry',reserve=reserve)

if __name__ == '__main__':
    unittest.main()
