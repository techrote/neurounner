"""Focused DREAMGRAFT-002 corpus, causal-control and likelihood tests."""
import unittest
from types import SimpleNamespace
from experiments.memplant.dreamgraft002.source import make_cases,HEAT,VIBRATION,PAIR_ANSWERS
from experiments.memplant.dreamgraft002.prefixes import QUESTIONS

class CorpusTests(unittest.TestCase):
    def test_factorial_group_disjoint_sources(self):
        xs=make_cases()
        self.assertEqual(len(xs),128)
        self.assertEqual([sum(x["split"]==k for x in xs) for k in ("train","validation","sealed")],[80,24,24])
        d={x["case_id"]:x for x in xs}
        for x in xs:
            h=d[x["heat_twin"]];v=d[x["vibe_twin"]]
            self.assertEqual((x["family"],x["vibration"]),(h["family"],h["vibration"]))
            self.assertNotEqual(x["heat"],h["heat"])
            self.assertEqual(x["heat"],v["heat"])
            self.assertNotEqual(x["vibration"],v["vibration"])
            self.assertEqual(x["source"].count("hardens"),1)
            self.assertEqual(x["source"].count("softens"),1)
            self.assertEqual(x["source"].count("floats"),1)
            self.assertEqual(x["source"].count("sinks"),1)
        self.assertEqual(len(set(x["source_sha256"] for x in xs)),128)
    def test_numeric_prompt_does_not_contain_donor_story(self):
        for q in QUESTIONS:
            self.assertIn("hardens",q)
            self.assertIn("floats",q)
            for x in make_cases():
                self.assertNotIn(x["source"],q)
                self.assertNotIn(x["case_id"],q)

try:
    import torch
    from torch import nn
    from experiments.memplant.dreamgraft002.evaluate import (
        two_bit_ridge,continuous_prefix,batch_four_choices)
    from experiments.memplant.dreamgraft001.bridge import answer_loss
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"PyTorch unavailable")
class ReceiverTests(unittest.TestCase):
    def test_two_output_readout_and_heat_vibration_twin(self):
        torch.manual_seed(31)
        xs=make_cases();x=torch.randn(128,2048)
        scores,stats=two_bit_ridge(x,xs)
        self.assertEqual(tuple(scores.shape),(128,2))
        self.assertEqual(stats["sealed"]["n"],24)
        self.assertTrue(torch.isfinite(scores).all())
        prefixes={f"class_{i}":torch.randn(1,8,2048) for i in range(4)}
        a,src,_=continuous_prefix(prefixes,scores,104,xs,"heat_swap",1)
        self.assertEqual(tuple(a.shape),(1,8,2048))
        self.assertNotEqual(xs[src]["heat"],xs[104]["heat"])
        self.assertEqual(xs[src]["vibration"],xs[104]["vibration"])
        a,src,_=continuous_prefix(prefixes,scores,104,xs,"vibration_swap",1)
        self.assertEqual(xs[src]["heat"],xs[104]["heat"])
        self.assertNotEqual(xs[src]["vibration"],xs[104]["vibration"])
    def test_batched_four_candidate_likelihood_matches_teacher_forced_loss(self):
        torch.manual_seed(14)
        class Tokenizer:
            eos_token_id=1
            def __call__(self,txt,add_special_tokens=False,return_tensors="pt"):
                words={"hardens":10,"softens":11,"floats":12,"sinks":13}
                return {"input_ids":torch.tensor([[words[w] for w in txt.split()]])}
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.embed=nn.Embedding(32,16);self.head=nn.Linear(16,32)
            def get_input_embeddings(self):return self.embed
            def forward(self,inputs_embeds,attention_mask,use_cache,return_dict):
                return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(dim=1)))
        tok=Tokenizer();receiver=Toy().eval()
        prompt=torch.tensor([[3,4,5]],dtype=torch.long)
        values=batch_four_choices(receiver,tok,prompt,None)
        self.assertEqual(len(values),4)
        for k,phrase in enumerate(PAIR_ANSWERS):
            ids=tok(phrase)["input_ids"]
            expected=-float(answer_loss(receiver,prompt,ids))*ids.numel()
            self.assertAlmostEqual(values[k],expected,places=4)

if __name__=="__main__":
    unittest.main()
