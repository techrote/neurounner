"""Independent focused regressions for synthetic goal/recovery curriculum."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import recovery_curriculum as rc


class CurriculumTests(unittest.TestCase):
    def test_oriented_teacher_is_host_legal(self):
        for seed in (1000, 1003, 1017):
            w=rc.make_world(seed);d=rc.distance_field(w)
            pos=w['start'];heading=0;visited={pos};history=[]
            for _ in range(64):
                moves,fresh,legal=rc.legal_state(w,pos,heading,visited,history)
                chosen,neg=rc.oracle(w,pos,heading,visited,history,d,legal)
                self.assertIn(chosen,legal)
                self.assertFalse(chosen in neg)
                if chosen=='STOP':break
                pos,heading=rc.apply(w,pos,heading,visited,history,chosen)
                self.assertNotIn(pos,w['obstacles']|w['protected'])
                visited.add(pos)

    def test_hard_negatives_and_train_lineage_without_privileged_prompt(self):
        rows,worlds,hard,_,_=rc.teacher_data((1000,1001),budget=48)
        self.assertGreater(hard['legal_distractors'],0)
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'a.zip';audit=rc.write_train_export(rows,worlds,target)
            self.assertTrue(audit['duplicates_removed']>=0)
            with zipfile.ZipFile(target) as z:
                m=json.loads(z.read('manifest.json'))
                msgs=z.read('train.messages.jsonl').splitlines()
                prov=z.read('train.provenance.jsonl').splitlines()
                neg=z.read('train.hard_negatives.jsonl').splitlines()
            self.assertEqual(len(msgs),len(prov))
            self.assertEqual(len(neg),len(msgs))
            self.assertEqual(m['row_count'],len(msgs))
            self.assertEqual(m['world_count'],2)
            for i,raw in enumerate(msgs):
                message=json.loads(raw)['messages'];obs=json.loads(message[1]['content'])
                self.assertNotIn('geodesic',obs)
                self.assertNotIn('world_seed',obs)
                self.assertIn(message[2]['content'],obs['legal'])
                self.assertEqual(json.loads(prov[i])['split'],'TRAIN')
                self.assertEqual(json.loads(neg[i])['correct_action'],message[2]['content'])

    def test_train_export_byte_reproducibility(self):
        rows,worlds,_,_,_=rc.teacher_data((1002,1004),budget=64)
        with tempfile.TemporaryDirectory() as tmp:
            a=rc.write_train_export(rows,worlds,Path(tmp)/'a.zip')
            b=rc.write_train_export(rows,worlds,Path(tmp)/'b.zip')
            self.assertEqual(a['sha256'],b['sha256'])
            self.assertEqual(a['rows'],b['rows'])

    def test_seed_overlap_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError,'leakage'):
                rc.run(Path('not-read-on-overlap.zip'),
                       output,train_seeds=(1000,),eval_seeds=(1000,))
            self.assertFalse(output.exists())

    def test_bad_source_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'forged.zip';source.write_bytes(b'not-a-real-source')
            with self.assertRaisesRegex(ValueError,'pin mismatch'):
                rc.run(source,Path(tmp)/'run',train_seeds=(1000,),eval_seeds=(2000,))


if __name__=='__main__':unittest.main()
