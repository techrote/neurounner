"""Isolation, deduplication, legality, and fail-closed tests for size-matched v4."""
from pathlib import Path
import tempfile
import unittest
from recovery_curriculum import make_world, legal_state, observation
from topology_control_v4 import (sample_world_balanced, run, verify_parents,
                                  _summarize, SCHEMA)
from domain_recovery_v3 import world_digest, make_slalom_world


class MatchedTopologyTests(unittest.TestCase):
    def row(self, group, value):
        return {'world_group_id':group,'observable_sha256':str(value),
          'label':'STOP','observation':{'legal':['STOP'],'tick':0},
          'student_action':'STOP','was_student_teacher_disagreement':False,
          'repeated_oriented_cell':False}

    def test_exact_count_deterministic_and_all_groups(self):
        rows=[self.row(f'original_v1:{i}',f'{i}-{j}') for i in range(24) for j in range(8)]
        selected=sample_world_balanced(rows,97)
        self.assertEqual(len(selected),97)
        self.assertEqual(len(set(r['observable_sha256'] for r in selected)),97)
        self.assertEqual(len(set(r['world_group_id'] for r in selected)),24)
        self.assertEqual([r['observable_sha256'] for r in selected],
          [r['observable_sha256'] for r in sample_world_balanced(list(reversed(rows)),97)])

    def test_reject_duplicate_and_illegal_or_overflow(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            sample_world_balanced([self.row('a','x'),self.row('b','x')],2)
        bad=self.row('a','x');bad['label']='STRAIGHT'
        with self.assertRaisesRegex(ValueError,'illegal'):
            sample_world_balanced([bad],1)
        with self.assertRaisesRegex(ValueError,'sample size'):
            sample_world_balanced([self.row('a','x'),self.row('b','y')],1)

    def test_partition_overlap_before_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'not-created'
            with self.assertRaisesRegex(ValueError,'TRAIN/EVAL'):
                run(target,source=(1000,),original_only=(2300,2301),
                    mixed_original=(2300,),mixed_barrier=(7300,),
                    original_eval=(2300,),barrier_eval=(6300,),slalom_eval=(9300,),budget=48)
            self.assertFalse(target.exists())

    def test_world_count_mismatch_and_unshared_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'world-group counts'):
                run(Path(temp)/'no',original_only=(2300,),mixed_original=(2300,),mixed_barrier=(7300,))
            with self.assertRaisesRegex(ValueError,'subset'):
                run(Path(temp)/'no2',source=(1000,),original_only=(2300,2301),
                     mixed_original=(2300,2302),mixed_barrier=(),
                     original_eval=(),barrier_eval=(),slalom_eval=(),budget=48)

    def test_slalom_budget_cannot_make_success_impossible(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'evaluation goal infeasible'):
                run(Path(temp)/'no',budget=128,source=(1000,),
                    original_only=(2300,2301),mixed_original=(2300,),
                    mixed_barrier=(7300,),original_eval=(),barrier_eval=(),slalom_eval=(9300,))

    def test_source_pins_and_eval_world_reachability(self):
        verify_parents()
        for seed in (9300,9301):
            w=make_slalom_world(seed)
            self.assertEqual(world_digest(w),world_digest(make_slalom_world(seed)))
            self.assertNotEqual(world_digest(w),world_digest(make_world(seed)))

if __name__=='__main__':unittest.main()
