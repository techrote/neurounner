"""Source-only guards for D014/D015, including reject-on-mutation provenance.
No original parent ZIP required to run this fast test module.
"""
import copy
import unittest
import numpy as np

from .franken_source_distill import (
    ACTIONS, ALL, FAMILIES, PINS, SCHEMA, Q8, LLAMA,
    convert_choice, graph_features, softmax, encode, train, evaluate,
)

class RealSourceChecks(unittest.TestCase):
    def entry(self):
        legal = ['STRAIGHT','LEFT','RIGHT','BRANCH','STOP']
        ps = {a:0.0 for a in ACTIONS}
        ps.update({'STRAIGHT':0.1,'LEFT':0.1,'RIGHT':0.1,'BRANCH':0.6,'STOP':0.1})
        return {'status':'ok','http_status':200,'state':{'seed':570,'tick':0},
            'choice':'BRANCH', 'parsed':{'probabilities':ps},
            'observation':{'family':'architecture','heading':0,'tick':0,
                'x':7,'y':48,'legal':legal,'local_5x5':['.....','.....','..@..','.....','.....'],
                'last_actions':[]}}

    def test_graph_fingerprint_is_bounded(self):
        a=graph_features(list(range(24)))
        self.assertEqual(a.shape,(26,))
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a[24],0)
        with self.assertRaises(ValueError): graph_features([24])
        with self.assertRaises(ValueError): graph_features([-1])

    def test_all_source_variants_have_exact_unique_pins(self):
        self.assertEqual(len(PINS),10)
        self.assertEqual(len(ALL),11)
        self.assertEqual(len({v[0] for v in PINS.values()}),10)
        self.assertTrue(all(len(v[1])==64 for v in PINS.values()))
        self.assertEqual(len(FAMILIES),3)

    def test_illegal_and_wrong_source_identity_are_rejected(self):
        entry=self.entry()
        args=('branchwitch',14183687090,'0'*64,list(range(24)))
        row=convert_choice(entry,*args)
        self.assertEqual(row['schema'],SCHEMA)
        self.assertEqual(row['producer_model_sha256'],Q8)
        self.assertEqual(row['producer_llama_commit'],LLAMA)
        self.assertEqual(encode(row).shape,(129,))
        bad=copy.deepcopy(entry);bad['choice']='RETRACE'
        with self.assertRaises(ValueError):convert_choice(bad,*args)
        bad=copy.deepcopy(entry);bad['state']['seed']=571
        with self.assertRaises(ValueError):convert_choice(bad,*args)
        bad=copy.deepcopy(entry);bad['parsed']['probabilities']['LEFT']=.2
        with self.assertRaises(ValueError):convert_choice(bad,*args)

    def test_real_bounded_optimizer_updates_and_legal_scoring(self):
        row=convert_choice(self.entry(),'branchwitch',14183687090,'0'*64,list(range(24)))
        records=[row]*5
        x=np.stack([encode(r) for r in records])
        y=np.stack([[r['probabilities'][a] for a in ACTIONS] for r in records]).astype(np.float32)
        mask=np.stack([[a in r['observation']['legal'] for a in ACTIONS] for r in records])
        student,fit=train(x,y,mask,16,3,42,.18)
        self.assertEqual(fit['updates'],3)
        self.assertGreater(fit['parameters'],100)
        predicted=evaluate(student,records,x,y,mask,np.array([0,1]))
        self.assertEqual(len(predicted),2)
        self.assertTrue(all(np.isfinite(q['kl']) for q in predicted))
        self.assertAlmostEqual(float(np.sum(softmax(student.scores(x[0])))),1.0)
        with self.assertRaises(ValueError):train(x,y,mask,16,1000,42,.18)

if __name__=='__main__':unittest.main()
