"""MZ-004 source-only assertions: immutable model identity and actual trainable scope."""
import unittest
from .run import MODEL, PIN, SCHEMA, last_block_trainable_names, minibatches
from experiments.driv3r.model_zoo.mamba130.counterfactual import (
    samples, paired_switch_metrics, raw_features
)
from experiments.driv3r.model_zoo.mamba130.world import World, ACTIONS


class Pythia70SourceTests(unittest.TestCase):
    def test_immutable_upstream_and_schema(self):
        self.assertEqual(MODEL, 'EleutherAI/pythia-70m-deduped')
        self.assertEqual(PIN, 'e93a9faa9c77e5d09219f6c868bfc7a1bd65593c')
        self.assertEqual(SCHEMA, 'MZ004-PYTHIA70-GAME-v1')

    def test_last_block_selection_excludes_all_other_layers(self):
        names=[
            'gpt_neox.layers.3.attention.dense.weight',
            'gpt_neox.layers.5.attention.query_key_value.weight',
            'gpt_neox.layers.5.attention.dense.weight',
            'gpt_neox.layers.5.attention.dense.bias',
            'gpt_neox.layers.5.post_attention_layernorm.weight',
            'gpt_neox.layers.5.post_attention_layernorm.bias',
            'gpt_neox.layers.5.mlp.dense_4h_to_h.weight',
            'gpt_neox.final_layer_norm.weight',
            'embed_out.weight'
        ]
        picked=last_block_trainable_names(names,6)
        self.assertEqual(len(picked),4)
        self.assertTrue(all(n.startswith('gpt_neox.layers.5.') for n in picked))
        self.assertNotIn('gpt_neox.layers.5.attention.query_key_value.weight',picked)
        with self.assertRaisesRegex(RuntimeError, 'unexpected'):
            last_block_trainable_names(['gpt_neox.layers.4.attention.dense.weight'],6)
        with self.assertRaisesRegex(ValueError,'layer'):
            last_block_trainable_names(names,0)

    def test_reproducible_same_minibatches(self):
        batches=minibatches(80,48,8,8111)
        self.assertEqual(batches,minibatches(80,48,8,8111))
        self.assertEqual(len(batches),48)
        self.assertTrue(all(len(set(b))==8 and min(b)>=0 and max(b)<80 for b in batches))
        self.assertNotEqual(batches,minibatches(80,48,8,8112))
        with self.assertRaises(ValueError): minibatches(4,10,8,8111)

    def test_goal_swap_and_seed_isolation(self):
        tr=samples(range(400,406),8)
        de=samples(range(900,904),8)
        te=samples(range(1200,1204),8)
        self.assertFalse({r[2] for r in tr}&{r[2] for r in de})
        self.assertFalse({r[2] for r in de}&{r[2] for r in te})
        self.assertEqual(len(raw_features(tr[0][0])),114)
        self.assertGreater(paired_switch_metrics(te,[r[1] for r in te])['changed_target_pairs'],20)

    def test_host_legality_unchanged(self):
        w=World.make(31)
        first=w.legal()
        self.assertTrue(first)
        self.assertTrue(set(first).issubset(set(ACTIONS)))
        for a in ACTIONS:
            if a not in first:
                with self.assertRaisesRegex(ValueError,'illegal'):
                    w.apply(a)
                break


if __name__=='__main__':
    unittest.main()
