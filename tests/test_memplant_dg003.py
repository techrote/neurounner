"""DG003: tests for withheld causal mechanisms and counterfactual attribution."""
import unittest
from types import SimpleNamespace

from experiments.memplant.dreamgraft002.source import make_cases
from experiments.memplant.dreamgraft003.interactions import (
    cause_truth, paired_source, judgement, TASKS, CONDITIONS,
    one_two_answer_forward, condition_input
)

class MechanismTests(unittest.TestCase):
    def test_truth_tables_require_both_variables(self):
        combos=[(h,v) for h in ("hardens","softens") for v in ("floats","sinks")]
        truth={
            "xor":[True,False,False,True],
            "and":[True,False,False,False],
            "equal":[False,True,True,False]
        }
        for task,expected in truth.items():
            self.assertEqual([cause_truth(task,h,v) for h,v in combos],expected)
            self.assertIn("Answer only yes or no.",TASKS[task]["question"])
        with self.assertRaises(ValueError):
            cause_truth("invalid","hardens","floats")

    def test_counterfactual_source_swaps_preserve_other_rule(self):
        cases=make_cases()
        sealed=[(i,x) for i,x in enumerate(cases) if x["split"]=="sealed"]
        self.assertEqual(len(sealed),24)
        for i,x in sealed:
            for name in ("heat_swap","vibration_swap","both_swap"):
                j=paired_source(cases,i,name)
                y=cases[j]
                self.assertEqual(x["family"],y["family"])
                if name=="heat_swap":
                    self.assertNotEqual(x["heat"],y["heat"])
                    self.assertEqual(x["vibration"],y["vibration"])
                elif name=="vibration_swap":
                    self.assertEqual(x["heat"],y["heat"])
                    self.assertNotEqual(x["vibration"],y["vibration"])
                else:
                    self.assertNotEqual(x["heat"],y["heat"])
                    self.assertNotEqual(x["vibration"],y["vibration"])

    def test_mechanism_flip_count_per_factorial_family(self):
        cases=make_cases()
        group=[x for x in cases if x["family"]=="two-rule-26"]
        self.assertEqual(len(group),4)
        for task in TASKS:
            for changed in ("heat_swap","vibration_swap","both_swap"):
                count=0
                for x in group:
                    i=cases.index(x)
                    k=paired_source(cases,i,changed)
                    y=cases[k]
                    before=cause_truth(task,x["heat"],x["vibration"])
                    after=cause_truth(task,y["heat"],y["vibration"])
                    count+=before!=after
                expected=(
                    4 if task in ("xor","equal") and changed!="both_swap"
                    else 0 if task in ("xor","equal")
                    else 2 if changed!="both_swap"
                    else 2
                )
                self.assertEqual(count,expected,(task,changed))

    def test_no_interaction_labels_in_donor_source_or_prefix_training(self):
        cases=make_cases()
        for x in cases:
            for token in ("relay energizes","safety door","stabilizer activates",
                          "EXACTLY ONE of these two sensors","both ON or both OFF"):
                self.assertNotIn(token,x["source"])
        self.assertNotIn(" Yes"," hardens floats")


try:
    import torch
    from torch import nn
    from experiments.memplant.dreamgraft001.bridge import answer_loss
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"torch unavailable")
class ForwardTests(unittest.TestCase):
    def test_two_answer_batch_matches_non_batched_teacher_forcing(self):
        torch.manual_seed(7)
        class Tokenizer:
            eos_token_id=1
            def __call__(self,word,add_special_tokens=False,return_tensors="pt"):
                x={" Yes":11," No":12}[word]
                return {"input_ids":torch.tensor([[x]])}
        class Receiver(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding=nn.Embedding(32,16)
                self.head=nn.Linear(16,32)
            def get_input_embeddings(self):return self.embedding
            def forward(self,inputs_embeds,attention_mask,use_cache,return_dict):
                # Prefix influences all subsequent outputs.
                return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(dim=1)))
        tokenizer=Tokenizer()
        receiver=Receiver().eval()
        prompt=torch.tensor([[3,4,5,6]])
        prefix=torch.randn(1,8,16)
        for injected in (None,prefix):
            measured=one_two_answer_forward(receiver,tokenizer,prompt,injected)
            for j,word in enumerate((" Yes"," No")):
                answer=tokenizer(word)["input_ids"]
                loss=answer_loss(receiver,prompt,answer,injected)
                self.assertAlmostEqual(measured[j],-float(loss)*answer.numel(),places=4)

    def test_numeric_conditions_do_not_include_source_narrative(self):
        cases=make_cases()
        class Tokenizer:
            def __init__(self):self.calls=[]
            def apply_chat_template(self,messages,tokenize=True,
                                    add_generation_prompt=True,return_tensors="pt"):
                self.calls.append(messages[0]["content"])
                return torch.tensor([[1,2,3]])
        tokenizer=Tokenizer()
        vals=torch.randn(128,2)
        prototypes={f"class_{j}":torch.randn(1,8,2048) for j in range(4)}
        for condition in ("real_v","heat_swap","vibration_swap","both_swap",
                          "hard_gate","oracle_prefix","equal_prefix","no_prefix"):
            prompt,pref,src,exposure=condition_input(
                cases,104,condition,vals,prototypes,tokenizer,"xor")
            self.assertFalse(exposure["source_text_seen"])
            self.assertFalse(exposure["explicit_rule_words_seen"])
            self.assertNotIn(cases[104]["source"],tokenizer.calls[-1])
            self.assertEqual(prompt.shape,(1,3))
            if condition=="no_prefix":self.assertIsNone(pref)
            else:self.assertEqual(pref.shape,(1,8,2048))
        _,_,_,facts=condition_input(cases,104,"text_facts",vals,prototypes,tokenizer,"xor")
        self.assertTrue(facts["explicit_rule_words_seen"])

if __name__=="__main__":
    unittest.main()
