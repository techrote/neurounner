"""Independent D-2 synthetic fragility regression/negative controls."""
import io
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile
import zlib

from fragility_study import decode_synthetic_rgba_png, measure, run_archive, encode, sha


def png(ink, blocked=()):
    def chunk(k, d):
        return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(k+d)&0xffffffff)
    colors={(232,222,199,255):'ink',(108,65,71,255):'blocked'}
    rows = []
    for y in range(32):
        row=bytearray(b'\x00')
        for x in range(32):
            row.extend((232,222,199,255) if (x,y) in ink else
                       (108,65,71,255) if (x,y) in blocked else (24,33,43,255))
        rows.append(bytes(row))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',32,32,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b'')


class FragilityTests(unittest.TestCase):
    def test_single_cell_bridge_is_fragile(self):
        initial={(1,1),(2,1),(4,1),(5,1)}
        data=measure(initial,initial|{(3,1)})
        self.assertEqual((data['initial_components'],data['final_components'],data['new_articulation_cells']), (2,1,1))
        self.assertEqual(data['new_articulation_fraction'],1)

    def test_wider_redundant_bridge_removes_fragility(self):
        initial={(1,1),(1,2),(4,1),(4,2)}
        final=initial|{(2,1),(3,1),(2,2),(3,2)}
        self.assertEqual(measure(initial,final)['new_articulation_cells'],0)
        self.assertGreater(measure(initial,final)['solid_2x2_squares'],0)

    def test_disconnected_greedy_additions_not_a_topology_win(self):
        initial={(1,1),(6,1)}
        outcome=measure(initial,initial|{(2,1),(3,1)})
        self.assertEqual(outcome['component_reduction'],0)

    def test_material_erasure_refused(self):
        with self.assertRaises(ValueError):measure({(1,1)},set())

    def test_synthetic_png_roundtrip_permissions(self):
        data=png({(1,1),(2,1)},{(10,10)})
        ink,perm=decode_synthetic_rgba_png(data)
        self.assertEqual(ink,{(1,1),(2,1)})
        self.assertNotIn((10,10),perm)
        with self.assertRaises(ValueError):decode_synthetic_rgba_png(data+b'garbage')
        broken=bytearray(data);broken[-7]^=1
        with self.assertRaises(ValueError):decode_synthetic_rgba_png(bytes(broken))

    def test_explicit_qualification_never_promoted(self):
        # Archive is optional in an isolated source checkout; image fixtures
        # are never committed with the small D-2 research source.
        archived=Path(__file__).parent/'confirmation-provisional-evidence.zip'
        if not archived.exists():self.skipTest('original provisional ZIP not mounted')
        a=run_archive(archived); b=run_archive(archived)
        self.assertEqual(sha(encode(a)),sha(encode(b)))
        self.assertFalse(a['source_policy_verified'])
        self.assertEqual(a['qualification'],'INDEPENDENT_ARCHIVED_SYNTHETIC_MORPHOLOGY_ONLY')
        self.assertEqual(len(a['cases']),8)


if __name__=='__main__':unittest.main()
