"""D-STUDENT-005: synthetic frontier rescue and honest-stop ablation.

Research only. The 144 archived d1-2B parent receipts share seed 570. Neither
this module nor its source tests performs a parent forward or native gameplay.
Distils locally inspectable host-legal frontier/recovery advice into a 5,142-
parameter student while retaining D004 and D003 score priors.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .lab import ACTIONS, STYLES, DIRS, Game, feats
from .repair import adverse, load
from .trap_rebalance import CASES, masked_softmax, adam_update, rollout, novelty_scores

SCHEMA = 'cybersand.driv3r.distill.frontier-rescue/1'
SOURCE_HEAD = '60195b1c269abd8ff5aa8c3ca5b62569d1f10f5b'
TRAIN = range(113,177)
VALID = range(1025,1033)
TEST = range(4001,4017)
FRACTIONS = (.15,.30,.45)
BUDGET = 80

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(v): return json.dumps(v, sort_keys=True, separators=(',',':'), allow_nan=False)

def frontier_distance(g, pos, radius=4):
    """Oracle's own short-horizon frontier distance, NEVER model observation.

    Protected and blocked cells cannot be crossed. An already-painted target
    is not a frontier, but traversal through painted cells is permitted.
    """
    if not g.open(pos):return None
    q=deque([(pos,0)]);seen={pos}
    while q:
        p,d=q.popleft()
        if d>0 and p not in g.painted:return d
        if d>=radius:continue
        for dx,dy in DIRS:
            nxt=(p[0]+dx,p[1]+dy)
            if nxt not in seen and g.open(nxt):
                seen.add(nxt);q.append((nxt,d+1))
    return None

def frontier_scores(g, stagnant=False):
    """Local oracle favors near-term exits, style variation and dead-end rescue.

    Only the corpus producer uses actual global painted/frontier data. It must
    not leak into features or inference, so success measures compression.
    """
    obs=g.obs();legal=set(obs['legal']); scores=np.full(6,-1000.,dtype=np.float32)
    good=0
    for i,(act,delta) in enumerate((('STRAIGHT',0),('LEFT',-1),('RIGHT',1))):
        if act not in legal:continue
        p=g.target((g.heading+delta)%4)
        dist=frontier_distance(g,p)
        if dist is not None:good+=1
        scores[i]=(2.3 if p not in g.painted else -.9)+(1.2 if dist==1 else .4 if dist==2 else -.2 if dist is not None else -1.8)
        scores[i]-=.14*obs['last_actions'].count(act)
    if 'BRANCH' in legal:
        p,_=g.branch_target()
        dist=frontier_distance(g,p)
        if dist is not None:good+=1
        scores[3]=3.2+(0.7 if g.style in ('greyhound','branchwitch') else 0.)-.11*obs['last_actions'].count('BRANCH')
    if 'RETRACE' in legal:
        prev=g.history[-2]
        dist=frontier_distance(g,prev)
        scores[4]=(2.3 if good==0 else -1.4)+(1.0 if dist is not None and good==0 else 0.)
    if 'STOP' in legal:
        # Early STOP only when genuinely no legal movement/retrace; otherwise
        # acknowledge severe repeated squeeze loops after a long effort.
        trapped=len(legal)==1
        repeating=stagnant and g.tick>=24 and g.style!='greyhound'
        scores[5]=5.0 if trapped else (2.0 if repeating and good==0 else -7.0)
    return scores

def deadend_stagnant(g):
    if g.tick<24:return False
    trail=g.actions[-8:]
    return len(trail)>=8 and trail.count('RETRACE')>=4 and len(g.painted)/max(1,g.tick)<.42

def generate(models,limit=28):
    x=[];menus=[];oracle=[];caseids=[];hard=[];styles=[]
    for seed in TRAIN:
        for style in STYLES:
            for case in CASES:
                for driver in ('rebalance_004','restored_003','flat_continue'):
                    g=Game(seed,style)
                    rng=np.random.default_rng(seed*641+STYLES.index(style)*37+CASES.index(case)*13)
                    for _ in range(limit):
                        adverse(g,case);obs=g.obs();mask=np.array([a in obs['legal'] for a in ACTIONS],dtype=bool)
                        x.append(feats(obs));menus.append(mask)
                        stagnant=deadend_stagnant(g)
                        oracle.append(frontier_scores(g,stagnant));caseids.append(CASES.index(case))
                        hard.append(stagnant or ('RETRACE' in obs['legal'] and len(obs['legal'])<=2))
                        styles.append(STYLES.index(style))
                        if driver=='flat_continue':
                            menu=[a for a in obs['legal'] if a!='STOP'] or ['STOP']
                            action=menu[int(rng.integers(len(menu)))]
                        else:action=models[driver].predict(obs)
                        if action not in obs['legal']:raise ValueError('corpus action not host legal')
                        g.step(action)
                        if g.stopped:break
    xx=np.stack(x).astype(np.float32);ll=np.stack(menus);oz=np.stack(oracle)
    ci=np.asarray(caseids,dtype=np.uint8);hh=np.asarray(hard,dtype=bool);si=np.asarray(styles,dtype=np.uint8)
    if xx.shape[1]!=100 or not ll.any(1).all() or len(xx)<1000:raise ValueError('broken corpus')
    return xx,ll,masked_softmax(oz,ll),ci,hh,si

def make_targets(p4,p3,oracle,cases,hard,styles,strength):
    if not 0<=strength<=.6:raise ValueError('overstrong frontier ablation')
    if p4.shape!=p3.shape or p4.shape!=oracle.shape:raise ValueError('target mismatch')
    if np.any(~np.isfinite(p4)) or np.any(~np.isfinite(oracle)):raise ValueError('bad targets')
    # More frontier repair in dynamic traps and hard recovery, with style
    # rehearsal for the original expressive GREYHOUND/BRANCHWITCH modes.
    rate=np.where(cases==1,strength,np.where(cases==2,strength*.8,strength*.4))
    rate=np.where(hard,np.minimum(.6,rate*1.35),rate).astype(np.float32)[:,None]
    style=np.where(np.isin(styles,[1,2]),.13,.06).astype(np.float32)[:,None]
    target=(1-rate-style)*p4+style*p3+rate*oracle
    if (target<0).any():raise ValueError('negative target')
    return (target/target.sum(1,keepdims=True)).astype(np.float32)

def evaluate(models,seeds,budget=BUDGET):
    records=[]
    for seed in seeds:
        for style in STYLES:
            for case in CASES:
                for policy in models:
                    _,r=rollout(seed,style,case,policy,models,budget)
                    records.append(r)
    summary={}
    for case in CASES:
        for name in models:
            rows=[r for r in records if r['case']==case and r['controller']==name]
            summary[case+'/'+name]={k:float(np.mean([r[k] for r in rows])) for k in
                ('cells','branches','revisits','retraces','raw_invalid','deadends','recovery','decisions','repetition')}
    return records,summary

def score(validation):
    def m(c,n,k):return validation[c+'/'+n][k]
    # Selection is validation-only and deliberately rewards BOTH branch style
    # and coverage, rather than rewarding mere action legality or STOP.
    cells=0.;branches=0.;penalty=0.
    for case,w in (('normal',.2),('trap',.55),('squeeze',.25)):
        a=m(case,'frontier_005','cells');b=m(case,'rebalance_004','cells')
        cells+=w*(a-b)/max(1.,b)
        a=m(case,'frontier_005','branches');b=m(case,'rebalance_004','branches')
        branches+=w*(a-b)/max(1.,b)
        penalty+=w*max(0.,m(case,'rebalance_004','decisions')-m(case,'frontier_005','decisions')-3.)/80
    score=1.8*cells+.55*branches-1.1*penalty
    return {'score':float(score),'coverage_relative':float(cells),'branch_relative':float(branches),'premature_stop_penalty':float(penalty)}

def seed_block_delta(records):
    out={}
    for case in CASES:
        paired=[]
        for seed in TEST:
            old={r['style']:r for r in records if r['case']==case and r['seed']==seed and r['controller']=='rebalance_004'}
            new={r['style']:r for r in records if r['case']==case and r['seed']==seed and r['controller']=='frontier_005'}
            paired.append({k:float(np.mean([new[s][k]-old[s][k] for s in STYLES])) for k in ('cells','branches','revisits','decisions')})
        rng=np.random.default_rng(5005);stats={}
        for metric in ('cells','branches','revisits','decisions'):
            a=np.array([r[metric] for r in paired]);boots=a[rng.integers(len(a),size=(2000,len(a)))].mean(1)
            stats[metric]={'delta':float(a.mean()),'bootstrap_95pct':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
                           'wins':int(np.sum(a>0)),'ties':int(np.sum(a==0)),'losses':int(np.sum(a<0))}
        out[case]=stats
    return out

def gallery(models,out,case):
    im=Image.new('RGB',(5*192,4*218),(236,238,240));draw=ImageDraw.Draw(im)
    names=('repaired_002','restored_003','rebalance_004','frontier_005','flat_continue')
    for iy,(seed,style) in enumerate(zip((4001,4005,4009,4013),STYLES)):
        for ix,name in enumerate(names):
            g,_=rollout(seed,style,case,name,models,BUDGET)
            im.paste(g.png(),(ix*192,iy*218+24))
            draw.text((ix*192+3,iy*218+5),style+'/'+name,fill=(10,10,10))
    im.save(out)

def original_action_evidence(models, out):
    """Source W0-like action receipts and model-projected PNGs for independent replay."""
    receipts=[]; hashes={}
    out.mkdir(parents=True,exist_ok=True)
    for case in CASES:
        for seed,style in zip((4001,4005,4009,4013),STYLES):
            for label in ('rebalance_004','frontier_005'):
                g=Game(seed,style);m=models[label]
                for _ in range(BUDGET):
                    adverse(g,case);o=g.obs()
                    raw=m.predict(o,mask=False);chosen=m.predict(o,mask=True)
                    if chosen not in o['legal']:raise AssertionError('host legality failure')
                    receipts.append({'case':case,'seed':seed,'style':style,'controller':label,
                                     'tick':g.tick,'position':list(g.pos),'heading':g.heading,
                                     'local_5x5':o['local_5x5'],'legal':o['legal'],
                                     'raw':raw,'chosen':chosen})
                    g.step(chosen)
                    if g.stopped:break
                fn=f'original_{case}_{seed}_{label}.png'
                g.png().save(out/fn);hashes[fn]=sha(out/fn)
    return receipts,hashes

def main(argv=None):
    p=argparse.ArgumentParser()
    for arg in ('base','repair','restored','rebalance','source-metrics'):
        p.add_argument('--'+arg,required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--steps',type=int,default=160)
    a=p.parse_args(argv)
    if not 1<=a.steps<=300:raise ValueError('bounded steps 1..300')
    a.out.mkdir(parents=True,exist_ok=True)
    old=json.loads(a.source_metrics.read_text())
    for path,key in ((a.base,'base_sha256'),(a.repair,'repair_sha256'),(a.restored,'restored_sha256'),(a.rebalance,'student_sha256')):
        if sha(path)!=old[key]:raise ValueError('upstream checkpoint SHA mismatch: '+key)
    if old['new_original_parent_inference']!=0 or old['original_teacher_source_seed']!=570:
        raise ValueError('unexpected source-parent provenance')
    if not (TRAIN.stop<=VALID.start and VALID.stop<=TEST.start):raise ValueError('train/validation/test overlap')
    models={'repaired_002':load(a.repair),'restored_003':load(a.restored),'rebalance_004':load(a.rebalance)}
    xx,ll,oracle,cases,hard,styles=generate(models)
    p4=masked_softmax(models['rebalance_004'].scores(xx),ll)
    p3=masked_softmax(models['restored_003'].scores(xx),ll)
    np.savez_compressed(a.out/'frontier_corpus.npz',features=xx,legal=ll,frontier=oracle,base_004=p4,style_003=p3,
                        case_id=cases,hard_state=hard,style_id=styles)
    candidates=[]
    for i,fraction in enumerate(FRACTIONS):
        yy=make_targets(p4,p3,oracle,cases,hard,styles,fraction)
        if np.max(np.abs(yy.sum(1)-1))>1e-5 or np.any(yy[~ll]>1e-7):raise ValueError('illegal targets')
        model=load(a.rebalance)
        train=adam_update(model,xx,yy,steps=a.steps,lr=.0015,seed=5055,batch=160)
        fn=f'frontier_candidate_{fraction:.2f}.npz';model.save(a.out/fn)
        _,val=evaluate({**models,'frontier_005':model},VALID)
        candidates.append({'fraction':fraction,'checkpoint':fn,'sha256':sha(a.out/fn),'train':train,'validation':score(val),
                           'validation_summary':{k:v for k,v in val.items() if k.endswith('frontier_005')}})
    selected=max(candidates,key=lambda z:(z['validation']['score'],-z['fraction']))
    chosen=load(a.out/selected['checkpoint']);chosen.save(a.out/'frontier_student.npz')
    models={**models,'frontier_005':chosen,'flat_continue':None,'wall':None}
    traces,summary=evaluate(models,TEST)
    (a.out/'frontier_rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in traces))
    gallery(models,a.out/'frontier_trap_gallery.png','trap')
    gallery(models,a.out/'frontier_squeeze_gallery.png','squeeze')
    action_receipts,png_hashes=original_action_evidence(models,a.out/'original_pngs')
    (a.out/'frontier_action_receipts.jsonl').write_text(''.join(js(r)+'\n' for r in action_receipts))
    result={'schema':SCHEMA,'base_source_head':SOURCE_HEAD,'evidence_kind':'external_synthetic_game_student_optimizer',
            'new_parent_inference':0,'native_cybersand_executions':0,'parent_seed':570,'real_aerohound_forwards':0,
            'train_seeds':[TRAIN.start,TRAIN.stop-1],'validation_seeds':[VALID.start,VALID.stop-1],
            'test_seeds':[TEST.start,TEST.stop-1],'train_examples':int(len(xx)),
            'hard_negative_examples':int(hard.sum()),'train_updates':len(FRACTIONS)*a.steps,
            'checkpoint_001_sha256':sha(a.base),'checkpoint_002_sha256':sha(a.repair),
            'checkpoint_003_sha256':sha(a.restored),'checkpoint_004_sha256':sha(a.rebalance),
            'corpus_sha256':sha(a.out/'frontier_corpus.npz'),'candidates':candidates,'selection':selected['fraction'],
            'student_sha256':sha(a.out/'frontier_student.npz'),'test_episodes':len(traces),'test_summary':summary,
            'paired_seed_block':seed_block_delta(traces),'rollout_sha256':sha(a.out/'frontier_rollouts.jsonl'),
            'trap_gallery_sha256':sha(a.out/'frontier_trap_gallery.png'),
            'squeeze_gallery_sha256':sha(a.out/'frontier_squeeze_gallery.png'),
            'original_pngs_sha256':png_hashes,'action_receipts':len(action_receipts),
            'action_receipts_sha256':sha(a.out/'frontier_action_receipts.jsonl'),
            'limitation':'Global short-horizon frontier oracle labels are synthetic, model sees 100 local features only'}
    (a.out/'frontier_metrics.json').write_text(js(result)+'\n')
    print(js({'selected':selected['fraction'],'updates':result['train_updates'],'examples':len(xx),
              'checkpoint_sha256':result['student_sha256'],
              'summary':{c:{n:summary[c+'/'+n] for n in ('rebalance_004','frontier_005')} for c in CASES},
              'paired':result['paired_seed_block']}))
    return result

if __name__=='__main__':main()
