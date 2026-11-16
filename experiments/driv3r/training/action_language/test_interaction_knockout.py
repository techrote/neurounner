"""GAL-007 fixed-checkpoint ablation and adversarial replay."""
import json
import tempfile
import unittest
from pathlib import Path

from lab import FEATURES, HostRefusal
from interaction_ablation import InkTargetWorld, INTERACTIONS, FEATURE_MAP
from interaction_knockout import (SCHEMA, PANEL_NAMES, EXPECTED, masked_model,
    checkpoint, episode, paired, run, audit, AuditRefusal)


class KnockoutTests(unittest.TestCase):
    def test_frozen_optimizer_checkpoints(self):
        for arm in ('cell','sweep'):
            for nonlinear in (False,True):
                model,_,h=checkpoint(arm,nonlinear)
                self.assertEqual(h,EXPECTED[f'{arm}_{"pairwise" if nonlinear else "linear"}'])
                self.assertEqual(len(model.weights),len(FEATURE_MAP if nonlinear else FEATURES))

    def test_one_variable_at_a_time(self):
        original,_,_=checkpoint('sweep',True)
        for name in INTERACTIONS:
            candidate=masked_model(original,'off_'+name)
            differences=[x for x,(a,b) in enumerate(zip(original.weights,candidate.weights)) if a!=b]
            self.assertTrue(set(differences)<={FEATURE_MAP.index(name)})
            self.assertEqual(candidate.weights[:len(FEATURES)],original.weights[:len(FEATURES)])
        self.assertEqual(masked_model(original,'full').weights,original.weights)
        self.assertTrue(all(masked_model(original,'all_off').weights[FEATURE_MAP.index(name)]==0 for name in INTERACTIONS))
        with self.assertRaises(ValueError):masked_model(original,'off_ghost')
        with self.assertRaises(ValueError):masked_model(original,'separately_trained_linear')

    def test_host_material_and_stop_authority(self):
        m,_,_=checkpoint('sweep',True)
        for panel in ('full','all_off','off_clear_x_span'):
            w,r=episode(masked_model(m,panel),'sweep',410000,'fresh_independent',panel)
            self.assertLessEqual(r['ink'],32)
            self.assertIn('STOP',w.legal())
            self.assertTrue(all(w.material[w.idx(q)]==w.original[w.idx(q)] for q in w.protected))
            self.assertTrue(all(e['lease']==r['lease'] for e in w.trace))
        with self.assertRaises(HostRefusal):w.apply('INK_E_999',w.offer()[1])

    def test_unmatched_refused(self):
        base={'seed':3,'family':'empty','world_sha256':'a','reached_32':True,
              'ink':32,'decisions':11,'receipts_sha256':'first'}
        with self.assertRaises(ValueError):paired([base],[{**base,'seed':4}])

    def test_bounded_and_tamper_replay(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):run(d,(4,5),0)
            report=run(d,(5,5),0)
            self.assertEqual(report['schema'],SCHEMA)
            replay=audit(d)
            self.assertEqual(replay['replayed_episodes'],5*2*2*len(PANEL_NAMES))
            data=Path(d,'receipts.json')
            obj=json.loads(data.read_text())
            obj['fresh_independent:sweep:full'][0][0]['action_id']='INK_N_999'
            data.write_text(json.dumps(obj))
            with self.assertRaises(AuditRefusal):audit(d)

if __name__=='__main__':unittest.main()
