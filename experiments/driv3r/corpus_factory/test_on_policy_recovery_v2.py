"""Independent focused correctness tests for TRAIN-only on-policy corrections."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import on_policy_recovery_v2 as op
import recovery_curriculum as rc


class OnPolicyTests(unittest.TestCase):
    def test_exact_student_score_parity(self):
        rows=rc.teacher_data((1000,1002),budget=40)[0]
        vec,model,_=rc.fit(rows,epochs=5)
        for seed in (2200,2203):
            w=rc.make_world(seed);pos=w['start'];heading=0
            visited={pos};history=[];past=[]
            for tick in range(25):
                moves,fresh,legal=rc.legal_state(w,pos,heading,visited,history)
                obs=rc.observation(w,pos,heading,visited,past,legal,tick,192,'greebles')
                original=rc.decide('recovery',obs,visited,vec,model)
                fast=op.decide_model(obs,vec,model)
                self.assertEqual(fast,original)
                if original=='STOP':break
                pos,heading=rc.apply(w,pos,heading,visited,history,original)
                visited.add(pos);past.append(original)

    def test_cross_generator_has_reachable_goal_and_distinct_geography(self):
        for seed in (6000,6001,6007):
            w=op.make_cross_world(seed);original=rc.make_world(seed)
            self.assertGreaterEqual(w['shortest_goal'],55)
            self.assertNotEqual(w['obstacles'],original['obstacles'])
            self.assertNotIn(w['start'],w['obstacles']|w['protected'])
            self.assertNotIn(w['goal'],w['obstacles']|w['protected'])

    def test_host_applied_goal_terminal_and_legal_only(self):
        rows=rc.teacher_data((1000,1001),budget=48)[0]
        vec,model,_=rc.fit(rows,epochs=4)
        for w in (rc.make_world(4000),op.make_cross_world(6000)):
            for family in op.FAMILIES:
                r=op.goal_terminal_rollout(w,family,vec,model,80)
                self.assertLessEqual(r['turns'],80)
                self.assertTrue(all(p not in w['obstacles']|w['protected'] for p in r['path']))
                if r['goal_tick'] is not None:
                    self.assertEqual(r['path'][-1],w['goal'])
                    self.assertEqual(r['turns'],r['goal_tick'])

    def test_reject_leaked_world_group_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'result'
            with self.assertRaisesRegex(ValueError,'leakage'):
                op.run(dest,source_seeds=(1000,),correction_seeds=(2200,),
                       evaluation_seeds=(2200,),cross_seeds=(6000,),budget=64)
            self.assertFalse(dest.exists())

    def test_train_only_hash_and_counterexample_provenance(self):
        source=rc.teacher_data((1000,1001),budget=48)[0]
        vec,model,_=rc.fit(source,epochs=3)
        raw,worlds,_=op.collect_corrections((2200,2201),vec,model,48)
        good,rejected=op.dedup_corrections(raw,source)
        self.assertTrue(good)
        self.assertLessEqual(len(good),len(raw))
        self.assertIn('same_state_as_source_or_prior_correction',rejected)
        with tempfile.TemporaryDirectory() as tmp:
            zpath=Path(tmp)/'train.zip';op.export_corrections(good,worlds,zpath)
            with zipfile.ZipFile(zpath) as z:
                self.assertIsNone(z.testzip())
                manifest=json.loads(z.read('manifest.json'))
                rows=z.read('train.messages.jsonl').splitlines()
                provenance=z.read('train.provenance.jsonl').splitlines()
                negatives=z.read('train.hard_negatives.jsonl').splitlines()
                self.assertEqual(len(rows),len(provenance))
                self.assertEqual(len(rows),len(negatives))
                self.assertEqual(len(rows),manifest['count'])
                for name,digest in manifest['members_sha256'].items():
                    self.assertEqual(op.sha(z.read(name)),digest)
                for i,(line,src,negative) in enumerate(zip(rows,provenance,negatives)):
                    messages=json.loads(line)['messages'];obs=json.loads(messages[1]['content'])
                    label=messages[2]['content'];receipt=json.loads(src)
                    self.assertEqual(receipt['split'],'TRAIN')
                    self.assertEqual(receipt['index'],i)
                    self.assertIn(label,obs['legal'])
                    self.assertNotIn('geodesic',obs)
                    self.assertEqual(receipt['teacher_label'],label)
                    self.assertEqual(receipt['observable_sha256'],op.sha(json.dumps(
                        {k:v for k,v in obs.items() if k!='family'},sort_keys=True,
                        separators=(',',':')).encode()))
                    self.assertEqual(json.loads(negative)['teacher_action'],label)

    def test_original_and_correction_state_dedup(self):
        rows=rc.teacher_data((1000,),budget=32)[0]
        ob,label,seed,neg=rows[0]
        key=op.sha(json.dumps({k:v for k,v in ob.items() if k!='family'},
                              sort_keys=True,separators=(',',':')).encode())
        correction={'observation':dict(ob),'label':label,'seed':2200,
                    'observable_sha256':key,'negative_actions':neg}
        kept,dropped=op.dedup_corrections([correction,correction],rows)
        self.assertEqual(kept,[])
        self.assertEqual(dropped['same_state_as_source_or_prior_correction'],2)

    def test_tiny_repeatable_complete_source_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs=[Path(tmp)/'a',Path(tmp)/'b']
            outputs=[op.run(d,source_seeds=tuple(range(1000,1006)),
                correction_seeds=(2200,2201),evaluation_seeds=(4000,4001),
                cross_seeds=(6000,6001),budget=64) for d in dirs]
            self.assertEqual(outputs[0],outputs[1])
            self.assertEqual((dirs[0]/'report.json').read_bytes(),(dirs[1]/'report.json').read_bytes())
            self.assertEqual((dirs[0]/'train_only_corrections.zip').read_bytes(),
                             (dirs[1]/'train_only_corrections.zip').read_bytes())
            self.assertTrue(all(x['native_events']==0 for x in outputs))


if __name__=='__main__':unittest.main()
