#!/usr/bin/env python3
"""Safety, action-menu, split isolation and representation-ablation tests for v6."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from recovery_curriculum import (ACTIONS,apply,legal_state,make_world,observation)
from on_policy_recovery_v2 import make_cross_world
from transfer_orientation_v5 import make_vertical_comb_world
from domain_recovery_v3 import make_slalom_world
from spatial_memory_v6 import (candidate_targets,candidate_features,
    memory_for,observation_pack,model_action,run_policy,partitions,verify_sources,
    export_train,fit_ranker,state_key)

class SpatialMemoryTests(unittest.TestCase):
    def test_source_pins_match(self):
        verify_sources(Path(__file__).parent)

    def test_split_seed_collision_fails(self):
        with self.assertRaisesRegex(ValueError,'seed collision'):
            partitions(source=(1000,),train_original=(2600,),train_horizontal=(7600,),
              train_vertical=(8600,),dev_original=(1000,),dev_horizontal=(6600,),
              held_original=(4700,),held_horizontal=(6700,),held_slalom=(9700,))

    def test_short_budget_rejected_before_train(self):
        with self.assertRaisesRegex(ValueError,'minimum route budget'):
            partitions(budget=128)

    def test_exact_legal_targets_and_protection(self):
        for maker,seed in ((make_world,2600),(make_cross_world,7600),
                           (make_vertical_comb_world,8600),(make_slalom_world,9700)):
            w=maker(seed);pos=w['start'];head=0;visited={pos};history=[]
            for _ in range(10):
                moves,fresh,legal=legal_state(w,pos,head,visited,history)
                targets=candidate_targets(w,pos,head,visited,history,legal)
                self.assertEqual(set(targets),set(legal))
                for a,(target,heading) in targets.items():
                    if a not in ('RETRACE','STOP'):
                        self.assertNotIn(target,w['obstacles']|w['protected'])
                action=next((a for a in ('STRAIGHT','LEFT','RIGHT') if a in legal),None)
                if action is None:break
                pos,head=apply(w,pos,head,visited,history,action)
                self.assertEqual((pos,head),targets[action]);visited.add(pos)

    def test_long_history_features_actually_differ(self):
        obs={'x':10,'y':10,'heading':0,'goal_delta':[40,0],
             'budget_remaining':182,'goal_reached':False,'family':'greebles',
             'last_actions':[],'legal':['STRAIGHT','STOP'],
             'local_5x5':['.....']*5}
        target=((11,10),0)
        a=candidate_features(obs,'STRAIGHT',target,None)
        b=candidate_features(obs,'STRAIGHT',target,
              {'seen_total':8,'seen_oriented':3,'seen_8':2,'seen_32':7,
               'age':1,'is_two_step_return':1})
        self.assertTrue(set(b)-set(a));self.assertFalse(any(k.startswith('mem|') for k in a))

    def test_memory_counts_actual_path_not_hidden_grid(self):
        targets={'LEFT':((10,11),1),'RIGHT':((9,10),2)}
        x=memory_for(targets,[(9,10),(10,10),(9,10),(10,10)],
                          [((9,10),2),((10,10),0),((9,10),2),((10,10),0)])
        self.assertEqual(x['RIGHT']['seen_total'],2)
        self.assertEqual(x['RIGHT']['seen_oriented'],2)
        self.assertEqual(x['RIGHT']['age'],1)
        self.assertEqual(x['LEFT']['seen_total'],0)
        self.assertGreater(x['LEFT']['age'],100)

    def test_memory_state_key_family_and_tick_invariant(self):
        w=make_world(2600);pos=w['start'];h=0
        obs,targets,mem=observation_pack(w,pos,h,{pos},[],[],[pos],[(pos,h)],0,192,'greebles')
        b=dict(obs,family='pipework',tick=22,budget_remaining=170)
        self.assertEqual(state_key(obs,mem),state_key(b,mem))
        self.assertNotEqual(state_key(obs,mem),state_key(obs,{a:dict(v,age=2) for a,v in mem.items()}))

    def test_sparse_inference_matches_sklearn_ranking(self):
        obs={'x':10,'y':10,'heading':0,'goal_delta':[40,0],
             'budget_remaining':182,'goal_reached':False,'family':'greebles',
             'last_actions':[],'legal':['STRAIGHT','LEFT','STOP'],
             'local_5x5':['.....']*5}
        t={'STRAIGHT':((11,10),0),'LEFT':((10,9),3),'STOP':((10,10),0)}
        m={a:{'seen_total':i,'seen_oriented':i,'seen_8':i,'seen_32':i,
              'age':i*10,'is_two_step_return':i%2} for i,a in enumerate(obs['legal'])}
        feats=[candidate_features(obs,a,t[a],m[a]) for a in obs['legal']]
        vec=DictVectorizer(sort=True);X=vec.fit_transform(feats)
        extra=vec.transform([candidate_features(obs,'STRAIGHT',t['STRAIGHT'],m['STRAIGHT'])])
        trainX=np.vstack([X.toarray(),extra.toarray(),X.toarray()])
        y=[1,0,0,1,1,0,0]
        model=SGDClassifier(loss='log_loss',random_state=17,max_iter=20,tol=None).fit(trainX,y)
        scores=model.decision_function(X)
        slow=max(zip(obs['legal'],scores),key=lambda a:(float(a[1]),-ACTIONS.index(a[0])))[0]
        self.assertEqual(slow,model_action(obs,t,m,vec,model,True))

    def test_corrupt_train_export_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs={'legal':['STRAIGHT'],'family':'greebles'}
            row={'split':'HELDOUT','obs':obs,'label':'STRAIGHT'}
            with self.assertRaisesRegex(ValueError,'non-TRAIN'):
                export_train([row],[],[],Path(tmp)/'oops.zip')
            row['split']='TRAIN';row['label']='LEFT'
            with self.assertRaisesRegex(ValueError,'illegal'):
                export_train([row],[],[],Path(tmp)/'oops.zip')

    def test_missing_corpus_fails_before_overwrite(self):
        from spatial_memory_v6 import run
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                run(Path(tmp))

if __name__=='__main__':unittest.main()
