"""D-STUDENT-004: trap-aware style retention in external synthetic game only.

Train a 5,142-parameter float32 student on new on-policy states using the
D003 style anchor, D002 recovery anchor and synthetic host-derived novelty.
The original 144 d1-2B teacher receipts share seed 570; this module performs
NO new parent inference, CyberSand native gameplay or production integration.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .lab import ACTIONS,STYLES,DIRS,Game,feats
from .repair import adverse,load

SCHEMA='cybersand.driv3r.distill.trap-style-rebalance/1'
TRAIN=range(65,113)
SELECT=range(1017,1025)
TEST=range(3001,3017)
CASES=('normal','trap','squeeze')
BUDGET=80
FRACTIONS=(.15,.30,.45)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def masked_softmax(logits,legal,temp=1.):
    if logits.ndim!=2 or logits.shape!=legal.shape or logits.shape[1]!=len(ACTIONS):
        raise ValueError('bad action tensor')
    if not np.isfinite(logits).all() or not legal.any(1).all() or temp<=0:
        raise ValueError('invalid logits/menu/temperature')
    z=np.where(legal,logits/temp,-1e5)
    z-=z.max(1,keepdims=True)
    p=np.exp(z)*legal
    return (p/p.sum(1,keepdims=True)).astype(np.float32)

def novelty_scores(g):
    """Synthetic local obstacle/visited-cell correction, not a source-model label."""
    o=g.obs();legal=set(o['legal'])
    s=np.full(6,-1000.,dtype=np.float32);fresh=0
    for i,(a,delta) in enumerate((('STRAIGHT',0),('LEFT',-1),('RIGHT',1))):
        if a not in legal:continue
        hd=(g.heading+delta)%4;p=g.target(hd)
        novel=p not in g.painted;fresh+=int(novel)
        exits=sum(g.open((p[0]+dx,p[1]+dy)) and (p[0]+dx,p[1]+dy) not in g.painted for dx,dy in DIRS)
        s[i]=(1.15 if novel else -1.8)+.22*min(exits,3)-.2*o['last_actions'].count(a)
    if 'BRANCH' in legal:
        fresh+=1
        s[3]=1.15+(.45 if g.style=='branchwitch' else 0.)-.16*o['last_actions'].count('BRANCH')
    if 'RETRACE' in legal:s[4]=2.5 if fresh==0 else -1.5
    if 'STOP' in legal:s[5]=4. if len(legal)==1 else -6.
    return s

def rollout(seed,style,case,which,models,budget=BUDGET):
    game=Game(seed,style)
    rng=np.random.default_rng(seed*1009+STYLES.index(style))
    raw_invalid=deadends=recovery=0
    for _ in range(budget):
        adverse(game,case)
        o=game.obs();legal=o['legal']
        if which in models and models[which] is not None:
            m=models[which]
            raw=m.predict(o,mask=False)
            raw_invalid+=int(raw not in legal)
            action=m.predict(o,mask=True)
        elif which=='flat_continue':
            offered=[a for a in legal if a!='STOP'] or ['STOP']
            action=offered[int(rng.integers(len(offered)))]
        elif which=='wall':
            action=next(a for a in ('LEFT','STRAIGHT','RIGHT','BRANCH','RETRACE','STOP') if a in legal)
        else:raise ValueError('unknown controller')
        if action not in legal:raise AssertionError('host action not legal')
        dead=not any(a in legal for a in ('STRAIGHT','LEFT','RIGHT','BRANCH'))
        deadends+=int(dead);recovery+=int(dead and action=='RETRACE')
        game.step(action)
        if game.stopped:break
    row={'seed':seed,'style':style,'case':case,'controller':which,'budget':budget,
         'cells':len(game.painted),'branches':game.branch_count,
         'revisits':game.revisits,'retraces':game.retraces,'raw_invalid':raw_invalid,
         'deadends':deadends,'recovery':recovery,'decisions':game.tick,
         'stop':game.stopped,'repetition':1-len(game.painted)/max(1,game.tick)}
    return game,row

def generate(models):
    features=[];menus=[];nov=[];case_ids=[]
    for seed in TRAIN:
        for style in STYLES:
            for case in CASES:
                for driver in ('repaired_002','restored_003','flat_continue'):
                    g=Game(seed,style)
                    rng=np.random.default_rng(seed*731+STYLES.index(style)*31+CASES.index(case)*11)
                    for _ in range(36):
                        adverse(g,case);o=g.obs()
                        features.append(feats(o))
                        menus.append(np.array([a in o['legal'] for a in ACTIONS],bool))
                        nov.append(novelty_scores(g));case_ids.append(CASES.index(case))
                        if driver=='flat_continue':
                            offered=[a for a in o['legal'] if a!='STOP'] or ['STOP']
                            action=offered[int(rng.integers(len(offered)))]
                        else:action=models[driver].predict(o)
                        if action not in o['legal']:raise AssertionError('illegal training action')
                        g.step(action)
                        if g.stopped:break
    x=np.stack(features).astype(np.float32);legal=np.stack(menus)
    p2=masked_softmax(models['repaired_002'].scores(x),legal)
    p3=masked_softmax(models['restored_003'].scores(x),legal)
    pn=masked_softmax(np.stack(nov),legal)
    return x,legal,p2,p3,pn,np.asarray(case_ids,dtype=np.uint8)

def targets(p2,p3,novelty,cases,fraction):
    if not 0<=fraction<=1:raise ValueError('invalid mixing fraction')
    trap=(cases==CASES.index('trap'))[:,None]
    squeeze=(cases==CASES.index('squeeze'))[:,None]
    recovery=np.where(trap,fraction,np.where(squeeze,fraction*.55,fraction*.17))
    exploration=np.where(trap,.16,np.where(squeeze,.10,.06))
    p=(1-recovery-exploration)*p3+recovery*p2+exploration*novelty
    if np.any(p<0) or not np.isfinite(p).all():raise ValueError('negative/nonfinite target')
    p/=p.sum(1,keepdims=True)
    return p.astype(np.float32)

def adam_update(model,x,y,steps=180,lr=.002,seed=413,batch=192):
    if not 1<=steps<=300 or x.ndim!=2 or x.shape[1]!=100 or y.shape!=(len(x),6):
        raise ValueError('invalid bounded training shape/steps')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(abs(y.sum(1)-1))>1e-4:
        raise ValueError('invalid targets')
    rng=np.random.default_rng(seed)
    ps=(model.w1,model.b1,model.w2,model.b2)
    m=[np.zeros_like(p) for p in ps];v=[r.copy() for r in m];loss=[]
    for step in range(1,steps+1):
        ix=rng.integers(len(x),size=batch);xx=x[ix];yy=y[ix]
        h=np.tanh(xx@model.w1+model.b1)
        z=h@model.w2+model.b2;z-=z.max(1,keepdims=True)
        pred=np.exp(z);pred/=pred.sum(1,keepdims=True)
        loss.append(float(-(yy*np.log(np.maximum(pred,1e-7))).sum(1).mean()))
        d=(pred-yy)/batch
        dw2=h.T@d;db2=d.sum(0);dh=(d@model.w2.T)*(1-h*h)
        dw1=xx.T@dh;db1=dh.sum(0)
        for i,(p,g) in enumerate(zip(ps,(dw1,db1,dw2,db2))):
            m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*(g*g)
            p-=lr*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
    return {'optimizer':'float32-Adam','updates':steps,
            'loss_initial':float(np.mean(loss[:10])),
            'loss_final':float(np.mean(loss[-20:])),
            'parameters':sum(p.size for p in ps)}

def evaluate(models,seeds):
    records=[]
    for case in CASES:
        for seed in seeds:
            for style in STYLES:
                for name in models:
                    _,row=rollout(seed,style,case,name,models)
                    records.append(row)
    summary={}
    for case in CASES:
        for name in models:
            rows=[r for r in records if r['case']==case and r['controller']==name]
            summary[case+'/'+name]={k:float(np.mean([r[k] for r in rows])) for k in
                ('cells','branches','revisits','retraces','raw_invalid',
                 'deadends','recovery','decisions','repetition')}
    return records,summary

def selection_score(candidate,restored,repaired):
    trap=candidate['trap']['cells']/max(1.,restored['trap']['cells'])
    trap_br=candidate['trap']['branches']/max(1.,restored['trap']['branches'])
    normal_cov=candidate['normal']['cells']/max(1.,restored['normal']['cells'])
    normal_br=candidate['normal']['branches']/max(1.,restored['normal']['branches'])
    penalty=max(0.,.97-normal_cov)*2+max(0.,.85-trap_br)+max(0.,.9-normal_br)
    score=(trap-1.)*2+(.3*candidate['trap']['cells']/max(1.,repaired['trap']['cells']))-penalty
    return {'score':float(score),'trap_coverage_ratio':float(trap),
        'trap_branch_ratio':float(trap_br),'normal_coverage_ratio':float(normal_cov),
        'normal_branch_ratio':float(normal_br),'penalty':float(penalty)}

def gallery(models,out,case):
    seeds=(3001,3005,3009,3013)
    labels=('original_001','repaired_002','restored_003','rebalance_004','flat_continue')
    im=Image.new('RGB',(5*192,4*218),(235,238,240));d=ImageDraw.Draw(im)
    for iy,(seed,style) in enumerate(zip(seeds,STYLES)):
        for ix,name in enumerate(labels):
            g,_=rollout(seed,style,case,name,models)
            im.paste(g.png(),(ix*192,iy*218+24))
            d.text((ix*192+3,iy*218+4),style+'/'+name,fill=(10,10,10))
    im.save(out)

def paired_deltas(records):
    out={}
    for case in CASES:
        seed_rows=[]
        for seed in TEST:
            old=[r for r in records if r['case']==case and r['seed']==seed and r['controller']=='restored_003']
            new=[r for r in records if r['case']==case and r['seed']==seed and r['controller']=='rebalance_004']
            seed_rows.append({k:float(np.mean([n[k]-o[k] for o,n in zip(old,new)]))
                    for k in ('cells','branches','revisits')})
        stats={};rng=np.random.default_rng(4004)
        for key in ('cells','branches','revisits'):
            ds=np.array([r[key] for r in seed_rows])
            boots=ds[rng.integers(len(ds),size=(4000,len(ds)))].mean(1)
            stats[key]={'delta':float(ds.mean()),
                'block_bootstrap_95pct':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
                'wins':int(sum(ds>0)),'ties':int(sum(ds==0)),'losses':int(sum(ds<0))}
        out[case]=stats
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--repair',type=Path,required=True)
    ap.add_argument('--restored',type=Path,required=True)
    ap.add_argument('--style-metrics',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=180)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    old=json.loads(args.style_metrics.read_text())
    for p,k in ((args.base,'base_001_sha256'),(args.repair,'repair_002_sha256'),(args.restored,'restored_003_sha256')):
        if sha(p)!=old[k]:raise ValueError('upstream checkpoint SHA mismatch '+k)
    if old['new_parent_inference']!=0 or old['parent_seed']!=570 or old['source_corpus_sha256']!='91c59e1a08ccfe4a1cbae6252988cfb0db1d43e9b7908cf2629a2f7f1460470c':
        raise ValueError('unreviewed parent/source provenance')
    models={'original_001':load(args.base),'repaired_002':load(args.repair),
            'restored_003':load(args.restored)}
    x,legal,p2,p3,novelty,cases=generate(models)
    np.savez_compressed(args.out/'trap_rebalance_corpus.npz',features=x,legal=legal,
        repair_distribution=p2,style_distribution=p3,novelty_distribution=novelty,case_id=cases)
    _,baseline=evaluate({**models,'flat_continue':None},SELECT)
    candidates=[]
    for fraction in FRACTIONS:
        y=targets(p2,p3,novelty,cases,fraction)
        if np.any(y[~legal]>1e-7):raise ValueError('illegal target probability mass')
        model=load(args.restored)
        training=adam_update(model,x,y,steps=args.steps)
        name=f'trap_candidate_{fraction:.2f}.npz';model.save(args.out/name)
        _,val=evaluate({**models,'rebalance_004':model},SELECT)
        candidate={c:val[c+'/rebalance_004'] for c in CASES}
        restored={c:baseline[c+'/restored_003'] for c in CASES}
        repaired={c:baseline[c+'/repaired_002'] for c in CASES}
        candidates.append({'fraction':fraction,'training':training,'checkpoint':name,
            'sha256':sha(args.out/name),'selection':selection_score(candidate,restored,repaired),
            'validation':candidate})
    selected=max(candidates,key=lambda c:(c['selection']['score'],-c['fraction']))
    winner=load(args.out/selected['checkpoint'])
    winner.save(args.out/'trap_rebalanced_student.npz')
    evaluated={**models,'rebalance_004':winner,'flat_continue':None,'wall':None}
    records,summary=evaluate(evaluated,TEST)
    (args.out/'trap_rebalance_rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in records))
    gallery(evaluated,args.out/'trap_rebalance_gallery.png','trap')
    gallery(evaluated,args.out/'trap_rebalance_normal_gallery.png','normal')
    result={'schema':SCHEMA,'evidence_kind':'actual_f32_student_optimizer_on_synthetic_W0_like_host',
     'new_original_parent_inference':0,'actual_aerohound_parent_forwards':0,
     'original_teacher_source_seed':570,'train_seeds':[TRAIN.start,TRAIN.stop-1],
     'select_seeds':[SELECT.start,SELECT.stop-1],'test_seeds':[TEST.start,TEST.stop-1],
     'steps_per_candidate':args.steps,'candidate_count':len(candidates),
     'training_rows':int(len(x)),'case_rows':{c:int(sum(cases==i)) for i,c in enumerate(CASES)},
     'corpus_sha256':sha(args.out/'trap_rebalance_corpus.npz'),
     'base_sha256':sha(args.base),'repair_sha256':sha(args.repair),
     'restored_sha256':sha(args.restored),'selected_candidate':selected['fraction'],
     'selected_validation':selected['selection'],'candidates':candidates,
     'heldout_episodes':len(records),'heldout_summary':summary,
     'paired_seed_block_deltas':paired_deltas(records),
     'student_sha256':sha(args.out/'trap_rebalanced_student.npz'),
     'original_parent_corpus_sha256':old['original_teacher_sha256'],
     'trajectory_sha256':sha(args.out/'trap_rebalance_rollouts.jsonl'),
     'trap_gallery_sha256':sha(args.out/'trap_rebalance_gallery.png'),
     'normal_gallery_sha256':sha(args.out/'trap_rebalance_normal_gallery.png'),
     'research_limit':'Synthetic W0-like host, not native CyberSand or faithful parent transfer'}
    (args.out/'trap_rebalance_metrics.json').write_text(js(result)+'\n')
    print(js({'selected':selected['fraction'],'optimizer_updates':len(FRACTIONS)*args.steps,
      'train_rows':len(x),'validation':selected['selection'],
      'test':{k:v for k,v in summary.items() if k.endswith('restored_003') or k.endswith('rebalance_004')},
      'checkpoint_sha256':result['student_sha256']}))

if __name__=='__main__':main()
