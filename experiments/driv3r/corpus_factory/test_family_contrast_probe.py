#!/usr/bin/env python3
"""Self-contained, exact-source TRAIN-only family-probe regression suite."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from family_contrast_probe import (SYSTEM, SOURCE_SCHEMA, canonical, emit,
                                   family_score, load_export, make_probes,
                                   probe_targets, sha)


def fixture(path):
    messages=[]; provenance=[]
    for i in range(18):
        family=('greebles','pipework','architecture')[i%3]
        tick=i//3
        obs={'x':8+tick,'y':48,'heading':0,'local_5x5':['.....','.....','..@..','.....','.....'],
             'legal':['STRAIGHT','LEFT','RIGHT','BRANCH','STOP'],
             'last_actions':[],'family':family,'tick':tick,
             'local_material_5x5':[['empty']*5 for _ in range(5)],
             'local_protected_5x5':['00000']*5,
             'goal_delta':[45-tick,0],'goal_reached':False,'budget_remaining':96-tick}
        label='LEFT' if i==1 else 'STRAIGHT'
        signature=sha(canonical([obs['local_5x5'],obs['local_material_5x5'],
                                 obs['local_protected_5x5'],obs['goal_delta'],
                                 obs['heading'],obs['family'],obs['legal'],
                                 obs['last_actions']]).encode())
        messages.append({'messages':[{'role':'system','content':SYSTEM},
                                     {'role':'user','content':canonical(obs)},
                                     {'role':'assistant','content':label}]})
        provenance.append({'index':i,'record_id':f'ep-{i%3}:{tick:04d}',
                           'world_group_id':'motif:100','input_sha256':signature,
                           'label':label,'evidence_class':'synthetic-host-heuristic'})
    blobs={}
    for k, rows in [('train.messages.jsonl',messages),('train.provenance.jsonl',provenance)]:
        blobs[k]=''.join(canonical(row)+'\n' for row in rows).encode()
    receipt={'schema':SOURCE_SCHEMA,'split':'train-only',
             'data_class':'synthetic-heuristic-supervision','native_game_records':0,
             'model_optimizer_steps':0,'source_manifest_sha256':'a'*64,
             'source_audit':{'audit':'PASS'},
             'counts':{'selected':18,'world_groups':1},
             'files':{k:{'sha256':sha(v),'bytes':len(v),'rows':len(v.splitlines())}
                      for k,v in blobs.items()}}
    blobs['export-manifest.json']=canonical(receipt).encode()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for k,v in blobs.items(): z.writestr(k,v)


class FamilyContrastTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source=Path(self.tmp.name)/'train.zip'
        fixture(self.source)
        self.source_bytes,self.manifest,self.rows=load_export(self.source)

    def test_matched_family_action_difference_is_not_counted_by_name_only(self):
        result=family_score(self.rows)
        self.assertEqual(result['source_examples'],18)
        self.assertEqual(result['matched_same_world_state_groups'],6)
        self.assertEqual(result['family_sensitive_action_groups'],1)
        self.assertEqual(result['identical_action_groups'],5)

    def test_counterfactual_triples_are_host_legal_and_nonexecuted(self):
        selected, messages, provenance=make_probes(self.rows,cap=4,per_world=4,selection_seed=8)
        self.assertEqual(len(selected),4)
        self.assertEqual(len(messages),12)
        for case in range(4):
            chosen=set()
            for j in range(case*3,case*3+3):
                m=messages[j]['messages']; o=json.loads(m[1]['content'])
                self.assertIn(m[2]['content'],o['legal'])
                self.assertFalse(provenance[j]['was_host_applied'])
                self.assertFalse(provenance[j]['has_native_receipt'])
                self.assertIn(provenance[j]['source_family'],('greebles','pipework','architecture'))
                chosen.add(m[2]['content'])
            self.assertEqual(len(chosen),3)

    def test_export_reproducible_and_immutable(self):
        root=Path(self.tmp.name)
        a,b=root/'one',root/'two'
        x=emit(self.source,a,cap=4,per_world=4,selection_seed=9,probes=True)
        y=emit(self.source,b,cap=4,per_world=4,selection_seed=9,probes=True)
        self.assertEqual(x,y)
        for name,inventory in x['files'].items():
            self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
            self.assertEqual(sha((a/name).read_bytes()),inventory['sha256'])
        with self.assertRaises(FileExistsError): emit(self.source,a)

    def test_corrupt_prompt_rejected_before_any_probe_emission(self):
        with zipfile.ZipFile(io.BytesIO(self.source_bytes)) as z:
            members={n:z.read(n) for n in z.namelist()}
        lines=members['train.messages.jsonl'].splitlines()
        row=json.loads(lines[0]);row['messages'][1]['content']='{"teacher_scores":"leak"}'
        lines[0]=canonical(row).encode()
        members['train.messages.jsonl']=b'\n'.join(lines)+b'\n'
        changed=Path(self.tmp.name)/'changed.zip'
        with zipfile.ZipFile(changed,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for k,v in members.items():z.writestr(k,v)
        with self.assertRaisesRegex(ValueError,'integrity'):
            load_export(changed)

    def test_illegal_actions_refused_and_size_bounded(self):
        self.assertIsNone(probe_targets({'legal':['STOP','RETRACE'],'tick':1}))
        with self.assertRaisesRegex(ValueError,'unbounded'):
            make_probes(self.rows,cap=4097)
        legal=['STRAIGHT','LEFT','RIGHT','BRANCH','STOP']
        self.assertEqual(probe_targets({'legal':legal,'tick':0})['architecture'],'LEFT')
        self.assertEqual(probe_targets({'legal':legal,'tick':8})['architecture'],'RIGHT')


if __name__=='__main__':unittest.main()
