import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from ingest_w0 import ingest
from corpus import SOURCE

class ImportW0Tests(unittest.TestCase):
    def fixture(self, file, illegal=False):
        receipt={'contract':SOURCE['common_w0_contract'],
                 'engine':'python-stdlib-baseline-not-native',
                 'mode':'random','seed':77,'family':'pipework',
                 'steps_executed':1,'trajectory':[[7,48],[7,48]],
                 'events':[{'observation':{'tick':0,'legal':['STOP'],
                                           'local_5x5':['.....','.....','..@..','.....','.....']},
                            'applied':'RETRACE' if illegal else 'STOP',
                            'raw_choice':'STOP','raw_legal':True,
                            'position_after':[7,48]}]}
        with zipfile.ZipFile(file,'w') as z:
            z.writestr('test/original-png-and-json/pipework-random-77.json',json.dumps(receipt))
            z.writestr('test/original-png-and-json/pipework-random-77.png',b'\x89PNG\r\n\x1a\n')

    def test_reference_only_and_original_digests(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);z=root/'original.zip';self.fixture(z)
            r=ingest(z,root/'reference')
            self.assertEqual(r['records'],1)
            self.assertEqual(r['events'],1)
            self.assertEqual(r['original_pngs'],1)
            self.assertEqual(r['split_admission'],'NONE: NEVER appended to TRAIN/DEV/heldout')
            row=json.loads((root/'reference/w0-reference.jsonl').read_text())
            self.assertTrue(row['not_for_training'])
            self.assertEqual(row['evidence_class'],'read-only-W0-inference-reference-not-training')
            self.assertEqual(row['applied_actions'],['STOP'])

    def test_refuse_false_host_legality(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);z=root/'malformed.zip';self.fixture(z,illegal=True)
            with self.assertRaisesRegex(ValueError,'host-illegal'):
                ingest(z,root/'reference')

if __name__=='__main__': unittest.main()
