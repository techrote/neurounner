"""D-STUDENT-006: learn honest STOP on exhausted synthetic squeeze worlds.

Faithful ONLY to the external 24x24 Game contract. A conservative full-map BFS / STOP-only-menu oracle provides
training labels and a diagnostic upper bound; the 100-input, 5,142-param MLP
never receives reachability, route history, or game state outside feats(obs).
All parent d1/Sprocket evidence remains historic; no new parent forward.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .lab import ACTIONS, STYLES, DIRS, Game, feats
from .repair import adverse, load
from .trap_rebalance import CASES, masked_softmax, adam_update

SCHEMA='cybersand.driv3r.distill.exhaustion-recovery/1'
TRAIN=range(177,225)
VALID=range(1041,1049)
TEST=range(5001,5017)
BUDGET=80
FRACTIONS=(.35,.60,.85)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)

def reachable_frontier(g):
    """Geometric frontier check; caller ALSO excludes STOP-only host menus."""
    seen={g.pos};queue=deque([g.pos]);count=0
    while queue:
        p=queue.popleft()
        if p not in g.painted:return 1
        for dx,dy in DIRS:
            q=(p[0]+dx,p[1]+dy)
            if q not in seen and g.open(q):
                seen.add(q);queue.append(q)
    return count

def corpus(student):
    x=[];menus=[];live=[];hard=[];case_ids=[];source=[];records=[]
    # D005 and legal random collect OFF-policy late states, including synthetic
    # near-dead-ends and negative examples where paint surrounds a viable route.
    for seed in TRAIN:
        for style in STYLES:
            for case in CASES:
                for driver in ('frontier_005','flat_continue'):
                    g=Game(seed,style)
                    rng=np.random.default_rng(seed*2003+STYLES.index(style)*29+CASES.index(case)*7)
                    for _ in range(BUDGET):
                        adverse(g,case);o=g.obs();legal=o['legal']
                        frontier=int(reachable_frontier(g)>0 and len(legal)>1)
                        f=feats(o)
                        x.append(f)
                        menus.append([a in legal for a in ACTIONS])
                        live.append(frontier>0)
                        # Negative controls: few visible local unpainted cells
                        # but frontier still reachable by traversal of paint.
                        hard.append(frontier>0 and sum(row.count('.') for row in o['local_5x5'])<=3)
                        case_ids.append(CASES.index(case));source.append(driver)
                        if driver=='flat_continue':
                            menu=[a for a in legal if a!='STOP'] or ['STOP']
                            action=menu[int(rng.integers(len(menu)))]
                        else:action=student.predict(o)
                        if action not in legal:raise AssertionError('unsafe generated action')
                        g.step(action)
                        if g.stopped:break
    x=np.stack(x).astype(np.float32)
    menus=np.array(menus,dtype=bool);live=np.array(live,dtype=bool)
    hard=np.array(hard,dtype=bool);case_ids=np.array(case_ids,dtype=np.uint8)
    assert len(x)>1000 and np.all(menus.any(1)) and x.shape[1]==100
    return x,menus,live,hard,case_ids

def balanced_indices(live,hard,case_ids,seed=6006):
    """Bounded stratification; never changes held-out distributions."""
    rng=np.random.default_rng(seed)
    pools=(np.where(~live)[0],np.where(hard)[0],np.where(live & ~hard)[0])
    if not all(len(p)>50 for p in pools):raise ValueError('insufficient exhausted/hard/live examples')
    n=len(live)
    a=int(n*.26);b=int(n*.30)
    return np.concatenate([rng.choice(p,size=c,replace=True) for p,c in zip(pools,(a,b,n-a-b))])[rng.permutation(n)]

def target_probs(prior,menu,live,fraction):
    if not 0.0<fraction<1.0:raise ValueError('bad fraction')
    if prior.shape!=menu.shape or len(live)!=len(menu):raise ValueError('dimension mismatch')
    if not (menu[:,5].all() and (prior[~menu]==0).all()):raise ValueError('bad prior menu')
    clean=prior.copy()
    clean[live,5]=0
    denom=clean[live].sum(axis=1,keepdims=True)
    clean[live]=np.divide(clean[live],denom,out=np.zeros_like(clean[live]),where=denom>0)
    # If STOP is the only action, cannot construct a live negative example.
    if np.any(denom==0):
        raise AssertionError('live frontier cannot have STOP-only menu')
    desired=clean
    desired[~live]=0
    desired[~live,5]=1
    p=((1-fraction)*prior+fraction*desired).astype(np.float32)
    p*=menu
    p/=p.sum(1,keepdims=True)
    return p

def evaluate_model(models,seeds,budget=BUDGET,record=False):
    rows=[];actions=[];pictures={}
    for seed in seeds:
        for style in STYLES:
            for case in CASES:
                for name,model in models.items():
                    g=Game(seed,style);raw_invalid=wrong_stop=stop_at_exhausted=0;exhaust_turn=None
                    snapshots=[]
                    for tick in range(budget):
                        adverse(g,case);o=g.obs();legal=o['legal'];f=int(reachable_frontier(g)>0 and len(legal)>1)
                        if f==0 and exhaust_turn is None:exhaust_turn=tick
                        if name=='oracle_stop' and f==0:action='STOP';raw='STOP'
                        else:
                            raw=model.predict(o,mask=False)
                            raw_invalid+=int(raw not in legal)
                            action=model.predict(o,mask=True)
                        if action not in legal:raise AssertionError('invalid selected action')
                        wrong_stop+=int(action=='STOP' and f>0)
                        stop_at_exhausted+=int(action=='STOP' and f==0)
                        if record:
                            actions.append({'seed':seed,'style':style,'case':case,'policy':name,
                                            'tick':tick,'position':list(g.pos),'heading':g.heading,
                                            'frontier_exists_oracle':f>0,'legal':legal,
                                            'local_5x5':o['local_5x5'],'raw':raw,'chosen':action})
                        g.step(action)
                        if g.stopped:break
                    row={'seed':seed,'style':style,'case':case,'policy':name,
                         'cells':len(g.painted),'branches':g.branch_count,'revisits':g.revisits,
                         'retraces':g.retraces,'decisions':g.tick,'stop':int(g.stopped),
                         'premature_stop':wrong_stop,'exhausted_stop':stop_at_exhausted,
                         'first_exhausted_turn':exhaust_turn if exhaust_turn is not None else -1,
                         'wasted_after_exhaustion':max(0,g.tick-(exhaust_turn+1)) if exhaust_turn is not None else 0,
                         'raw_invalid':raw_invalid}
                    rows.append(row)
                    if record and seed in (5001,5005,5009,5013):
                        fn=f'original_{case}_{seed}_{style}_{name}.png';pictures[fn]=g.png()
    return rows,actions,pictures

def summarize(rows):
    out={}
    for case in CASES:
        for name in sorted({r['policy'] for r in rows}):
            matches=[r for r in rows if r['case']==case and r['policy']==name]
            out[case+'/'+name]={k:float(np.mean([r[k] for r in matches])) for k in
               ('cells','branches','revisits','retraces','decisions','stop','premature_stop',
                'exhausted_stop','wasted_after_exhaustion','raw_invalid')}
    return out

def selection_score(v,name='recovery_006'):
    def m(case,n,k):return v[case+'/'+n][k]
    score=0
    for case,w in (('normal',.25),('trap',.30),('squeeze',.45)):
        a=m(case,name,'cells');b=m(case,'frontier_005','cells')
        score+=w*(a-b)*.12
        score+=w*(m(case,name,'branches')-m(case,'frontier_005','branches'))*.06
        score-=w*m(case,name,'premature_stop')*12
    score+=.45*(m('squeeze','frontier_005','wasted_after_exhaustion')-m('squeeze',name,'wasted_after_exhaustion'))
    return float(score)

def gallery(models,out,case):
    names=('frontier_005','recovery_006','oracle_stop')
    im=Image.new('RGB',(len(names)*192,4*218),(235,238,241));dr=ImageDraw.Draw(im)
    for row,(seed,style) in enumerate(zip((5001,5005,5009,5013),STYLES)):
        for col,name in enumerate(names):
            one,_,pics=evaluate_model({name:models[name]},[seed],record=True)
            target=pics.get(f'original_{case}_{seed}_{style}_{name}.png')
            if target is None:raise AssertionError('missing replay image')
            im.paste(target,(col*192,row*218+24))
            dr.text((col*192+2,row*218+4),f'{style} / {name}',fill=(12,12,12))
    im.save(out)

def main(argv=None):
    p=argparse.ArgumentParser()
    for k in ('base','metrics','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--steps',type=int,default=160)
    a=p.parse_args(argv)
    if not 1<=a.steps<=300:raise ValueError('bounded optimizer updates 1..300')
    old=json.loads(a.metrics.read_text())
    if digest(a.base)!=old['student_sha256']:raise ValueError('upstream D005 checkpoint mismatch')
    if old['new_parent_inference'] or old['native_cybersand_executions']:raise ValueError('invalid source type')
    assert not (set(TRAIN)&set(VALID) or set(TRAIN)&set(TEST) or set(VALID)&set(TEST))
    a.out.mkdir(parents=True,exist_ok=True)
    prior=load(a.base)
    xx,menus,live,hard,cases=corpus(prior)
    ix=balanced_indices(live,hard,cases)
    pp=masked_softmax(prior.scores(xx),menus)
    np.savez_compressed(a.out/'exhaustion_corpus.npz',features=xx,legal=menus,frontier_live=live,
                        hard_live=hard,case_id=cases,prior=pp,train_indices=ix)
    candidates=[]
    for fraction in FRACTIONS:
        yy=target_probs(pp,menus,live,fraction)
        m=load(a.base)
        train=adam_update(m,xx[ix],yy[ix],steps=a.steps,lr=.0014,seed=6606,batch=192)
        fn=f'exhaustion_candidate_{fraction:.2f}.npz';m.save(a.out/fn)
        vrows,_,_=evaluate_model({'frontier_005':prior,'recovery_006':m},VALID)
        val=summarize(vrows)
        candidates.append({'fraction':fraction,'checkpoint':fn,'sha256':digest(a.out/fn),
                           'optimizer':train,'validation_score':selection_score(val),'validation':val})
    selected=max(candidates,key=lambda c:(c['validation_score'],-c['fraction']))
    student=load(a.out/selected['checkpoint']);student.save(a.out/'exhaustion_student.npz')
    models={'frontier_005':prior,'recovery_006':student,'oracle_stop':prior}
    rows,actions,pics=evaluate_model(models,TEST,record=True)
    summary=summarize(rows)
    (a.out/'exhaustion_rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in rows))
    (a.out/'exhaustion_action_receipts.jsonl').write_text(''.join(js(r)+'\n' for r in actions))
    pngdir=a.out/'original_pngs';pngdir.mkdir(exist_ok=True)
    for fn,png in pics.items():png.save(pngdir/fn)
    # 2x4 gallery: same image evidence, no re-simulation needed.
    for case in ('squeeze','trap'):
        im=Image.new('RGB',(576,4*218),(235,238,241));dr=ImageDraw.Draw(im)
        for i,(seed,style) in enumerate(zip((5001,5005,5009,5013),STYLES)):
            for j,name in enumerate(models):
                im.paste(pics[f'original_{case}_{seed}_{style}_{name}.png'],(j*192,i*218+24))
                dr.text((j*192+2,i*218+4),f'{style}/{name}',fill=(9,9,9))
        im.save(a.out/f'exhaustion_{case}_gallery.png')
    result={'schema':SCHEMA,'source_d005_sha256':digest(a.base),'input_metrics_sha256':digest(a.metrics),
       'new_parent_model_forwards':0,'new_native_executions':0,'train_seeds':[TRAIN.start,TRAIN.stop-1],
       'validation_seeds':[VALID.start,VALID.stop-1],'test_seeds':[TEST.start,TEST.stop-1],
       'training_observations':len(xx),'train_exhausted_positive':int((~live).sum()),
       'train_hard_live_negative':int(hard.sum()),'balanced_observations':len(ix),
       'optimizer_updates':len(FRACTIONS)*a.steps,'candidates':candidates,
       'selected_fraction':selected['fraction'],'checkpoint_sha256':digest(a.out/'exhaustion_student.npz'),
       'test_episodes':len(rows),'test_summary':summary,
       'actions':len(actions),'action_sha256':digest(a.out/'exhaustion_action_receipts.jsonl'),
       'original_png_sha256':{k:digest(pngdir/k) for k in sorted(pics)},
       'warning':'Only 100 local features are available to the MLP; oracle BFS is synthetic teacher-only and not a native game API'}
    (a.out/'exhaustion_metrics.json').write_text(js(result)+'\n')
    print(js({'selected_fraction':selected['fraction'],'validation_score':selected['validation_score'],
              'checkpoint_sha256':result['checkpoint_sha256'],'examples':len(xx),
              'summary':{case:{name:summary[case+'/'+name] for name in models} for case in CASES}}))
    return result
if __name__=='__main__':main()
