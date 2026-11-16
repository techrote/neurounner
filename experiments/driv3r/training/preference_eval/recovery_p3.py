"""P3 synthetic host-legal recovery study. NO native Godot/real d1 inference/human ratings.

Trains a tiny conditional action ranker on legal one-step choices in independent
material/obstacle/protected worlds. Outcomes are external simulation proxy events.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, pathlib, struct, zlib

DIRS={'N':(0,-1),'E':(1,0),'S':(0,1),'W':(-1,0)}
ACTIONS=tuple('INK_'+d for d in DIRS)+tuple('LIFT_'+d for d in DIRS)+('STOP',)
FEATURES=('bias','ink','lift','stop','new','owned','xwest','east','turn_budget','dist_exit',
          'next_open','local_block','material_contact','heading_same')
SIZE=20

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def make_world(seed,kind):
    rng=random.Random(seed);y=8+rng.randrange(4)
    blocked={(x,0) for x in range(SIZE)}|{(x,SIZE-1) for x in range(SIZE)}|{(0,t) for t in range(SIZE)}|{(SIZE-1,t) for t in range(SIZE)}
    # An actual one-cell cul-de-sac; the first exit requires a legal pen lift onto owned ink.
    blocked|={(6,y),(5,y-1),(5,y+1)}
    if kind in ('protected','zigzag'):
        blocked|={(8,t) for t in range(3,17) if t != (5 if seed%2 else 15)}
    if kind in ('obstacles','zigzag'):
        blocked|={(9,t) for t in range(5,15) if t not in (7,12)}
    protected={(7,y-2),(7,y-3)} if kind in ('protected','zigzag') else {(10,y+2)}
    for _ in range(7):
        p=(rng.randrange(7,17),rng.randrange(2,17))
        if p not in blocked and p not in protected:blocked.add(p)
    original={(3,y),(4,y),(5,y)}
    assert not (blocked&original) and not (protected&original)
    return {'seed':seed,'kind':kind,'blocked':blocked,'protected':protected,'owned':original,
            'start':(5,y),'pos':(5,y),'previous':(4,y),'heading':'E','turn':0,'stopped':False,
            'ever_exited':False,'new_cells':0,'reversals':0,'events':[]}

def allowed(w):
    if w['stopped']:return ()
    opts=['STOP']
    for d,(dx,dy) in DIRS.items():
        q=(w['pos'][0]+dx,w['pos'][1]+dy)
        if q in w['blocked'] or q in w['protected'] or not (0<q[0]<SIZE-1 and 0<q[1]<SIZE-1):continue
        if q not in w['owned']:opts.append('INK_'+d)
        # Pen-up can travel only on Empty or this worm's own ink. No protected crossing.
        opts.append('LIFT_'+d)
    return tuple(sorted(opts))

def advance(w,a):
    opts=allowed(w)
    if a not in opts:raise ValueError('host-illegal action or already terminated')
    old=w['pos'];new=old
    if a!='STOP':
        dx,dy=DIRS[a[-1]];new=(old[0]+dx,old[1]+dy)
        if a.startswith('INK_'):
            if new in w['owned'] or new in w['protected'] or new in w['blocked']:
                raise ValueError('invalid painted cell')
            w['owned'].add(new);w['new_cells']+=1
        if new==w['previous']:w['reversals']+=1
        w['previous']=old;w['pos']=new;w['heading']=a[-1]
    else:w['stopped']=True
    w['turn']+=1
    if w['pos'][0]<=3 and w['pos']!=w['start']:w['ever_exited']=True
    w['events'].append({'t':w['turn'],'from':old,'to':new,'action':a,'legal_menu':opts,
                        'menu_sha256':digest({'t':w['turn']-1,'from':old,'legal':opts,
                                              'owned':sorted(w['owned']-({new} if a.startswith('INK_') else set()))}),
                        'new_material_cells':w['new_cells'],'escaped_culdesac':w['ever_exited']})

def features(w,a):
    x,y=w['pos'];dx,dy=DIRS[a[-1]] if a!='STOP' else (0,0)
    q=(x+dx,y+dy); ink=a.startswith('INK_');lift=a.startswith('LIFT_')
    neighbors=sum((q[0]+u,q[1]+v) in w['blocked'] or (q[0]+u,q[1]+v) in w['protected'] for u,v in DIRS.values())
    return (1.,float(ink),float(lift),float(a=='STOP'),float(ink and q not in w['owned']),
            float(q in w['owned']),float(q[0]<x),float(q[0]>x),1-w['turn']/40,
            float(q[0]<=3 and not w['ever_exited']),float(4-neighbors)/4,
            neighbors/4,float(any((q[0]+u,q[1]+v) in w['protected'] for u,v in DIRS.values())),
            float(a!='STOP' and a[-1]==w['heading']))

def score(weights,w,a):return sum(u*v for u,v in zip(weights,features(w,a)))

def choice(w,kind,weights=None,rng=None):
    menu=allowed(w)
    if kind=='random':return (rng or random).choice(menu)
    if kind=='random_low_stop':
        rng=rng or random
        if rng.random()<.04:return 'STOP'
        return rng.choice([a for a in menu if a!='STOP'] or ['STOP'])
    if kind=='recovery_rule':
        if not w['ever_exited'] and 'LIFT_W' in menu:return 'LIFT_W'
        return choice(w,'greedy',weights,rng)
    if kind=='greedy':
        # Strong transparent deterministic control, not a strawman uniform policy.
        return max(menu,key=lambda a:(int(a.startswith('INK_')),int(a=='LIFT_W' and not w['ever_exited']),
                                      int(a!='STOP' and a[-1]==w['heading']),-ACTIONS.index(a)))
    if kind in ('model','model_no_goal'):
        best=max(score(weights,w,a) for a in menu)
        winners=[a for a in menu if abs(score(weights,w,a)-best)<1e-10]
        return sorted(winners)[0]
    raise ValueError('unknown controller')

def run_episode(seed,kind,policy,weights=None,horizon=40):
    w=make_world(seed,kind);rng=random.Random(seed+320003)
    while w['turn']<horizon and not w['stopped']:
        a=choice(w,policy,weights,rng);advance(w,a)
    return w

def weak_label(w,a):
    # Only legal choices are labelled, with explicitly *algorithmic* weak supervision.
    # STOP is considered legal, never removed. No aesthetic or human preferences.
    f=features(w,a)
    return 2.5*f[4]+1.8*f[9]+.35*f[7]+.25*f[10]-.08*f[11]-.4*f[5]-1.6*f[3]

def make_training(seed_range):
    pairs=[]
    for seed in seed_range:
        kind=('open','obstacles','protected','zigzag')[seed%4]
        w=make_world(seed,kind)
        # Source of states: actual deterministic legal synthetic rollouts; no oracle teleport.
        for _ in range(13):
            menu=allowed(w)
            if len(menu)>=2:
                ordered=sorted(menu,key=lambda a:(-weak_label(w,a),a))
                good=ordered[0]
                for bad in ordered[1:]:
                    if weak_label(w,good)-weak_label(w,bad)>.15:
                        diff=tuple(x-y for x,y in zip(features(w,good),features(w,bad)))
                        pairs.extend(((diff,1),(tuple(-d for d in diff),0)))
            a=choice(w,'greedy')
            advance(w,a)
            if w['stopped']:break
    return pairs

def fit(pairs,steps=240,lr=.08,disabled=()):
    w=[0.]*len(FEATURES);m=[0.]*len(FEATURES);v=[0.]*len(FEATURES);history=[]
    for it in range(1,steps+1):
        grad=[.02*t for t in w];loss=0.
        for x,y in pairs:
            z=max(-30,min(30,sum(a*b for a,b in zip(w,x))));p=1/(1+math.exp(-z))
            loss-=y*math.log(max(1e-12,p))+(1-y)*math.log(max(1e-12,1-p))
            for j in range(len(w)):grad[j]+=(p-y)*x[j]/len(pairs)
        for j,g in enumerate(grad):
            if j in disabled:continue
            m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
            w[j]-=lr*(m[j]/(1-.9**it))/(math.sqrt(v[j]/(1-.999**it))+1e-8)
        if it in (1,40,120,steps):history.append([it,round(loss/len(pairs),6)])
    return {'weights':w,'adam_steps':steps,'training_pairs':len(pairs),'loss':history,
            'label_origin':'algorithmic local-action proxy; no human ratings','feature_names':FEATURES,
            'disabled_features':[FEATURES[i] for i in disabled]}

def audited_record(w,policy,weights_sha=None):
    trace={'schema':'pref-p3-external-synthetic/v1','seed':w['seed'],'kind':w['kind'],
       'controller':policy,'turns':w['turn'],'stopped':w['stopped'],
       'new_cells':w['new_cells'],'ever_exited':w['ever_exited'],'reversals':w['reversals'],
       'protected':sorted(w['protected']),'blocked':sorted(w['blocked']),
       'owned':sorted(w['owned']),'events':w['events'],'checkpoint_sha256':weights_sha,
       'native_receipts':0,'actual_resident_pretrained_models':0,'human_ratings':0}
    trace['trace_sha256']=digest(trace);return trace

def verify_receipt(receipt):
    if receipt['controller'] not in ('model','model_no_goal','greedy','recovery_rule','random','random_low_stop'):
        raise ValueError('unrecognised controller')
    if receipt['kind']!=('open','obstacles','protected','zigzag')[receipt['seed']%4]:
        raise ValueError('unrecognised World seed/kind pairing')
    w=make_world(receipt['seed'],receipt['kind'])
    for event in receipt['events']:
        advance(w,event['action'])
    reconstructed=audited_record(w,receipt['controller'],receipt['checkpoint_sha256'])
    if digest(reconstructed)!=digest(receipt):
        raise ValueError('changed action/menu/material/World/receipt hash')
    if any(p in w['blocked'] or p in w['protected'] for p in w['owned']):
        raise ValueError('painted protected or obstacle material')
    return True

def render_png(w,scale=8):
    # 100% derived from the synthetic material state; no AI image generation.
    rgb={(x,y):(100,116,139) for x,y in w['blocked']}
    rgb.update({p:(239,68,68) for p in w['protected']})
    rgb.update({p:(245,158,11) for p in w['owned']})
    n=SIZE*scale;raw=bytearray()
    for y in range(n):
        raw.append(0)
        for x in range(n):raw.extend(rgb.get((x//scale,y//scale),(14,19,29)))
    def chunk(tag,data):return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',n,n,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')

def heldout_agreement(pairs,weights):
    return {'pairs':len(pairs),'constant_label_baseline_correct':max(sum(y for _,y in pairs),sum(1-y for _,y in pairs)),
            'correct':sum((sum(w*v for w,v in zip(weights,x))>0)==bool(y) for x,y in pairs),
            'label_origin':'same predetermined weak-label formula, no human preference'}

def main(output):
    out=pathlib.Path(output);out.mkdir(parents=True,exist_ok=True)
    train_seeds=[i for i in range(1000,1080) if i%4!=3];test_seeds=range(9000,9024)
    pairs=make_training(train_seeds);head=fit(pairs);head_sha=digest(head)
    head_no_goal=fit(pairs,disabled=(FEATURES.index('dist_exit'),FEATURES.index('xwest')));no_goal_sha=digest(head_no_goal)
    (out/'p3-conditional-ranker.json').write_text(json.dumps(head,indent=2,sort_keys=True)+'\n')
    (out/'p3-no-goal-ranker.json').write_text(json.dumps(head_no_goal,indent=2,sort_keys=True)+'\n')
    rows=[];checks=[]
    for seed in test_seeds:
        for policy in ('model','model_no_goal','recovery_rule','greedy','random_low_stop','random'):
            world=run_episode(seed,('open','obstacles','protected','zigzag')[seed%4],policy,head_no_goal['weights'] if policy=='model_no_goal' else head['weights'])
            receipt=audited_record(world,policy,head_sha if policy=='model' else no_goal_sha if policy=='model_no_goal' else None)
            verify_receipt(receipt)
            stem=f'p3-{seed}-{policy}'
            (out/(stem+'.json')).write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
            (out/(stem+'.png')).write_bytes(render_png(world))
            rows.append({'seed':seed,'kind':world['kind'],'policy':policy,'new':world['new_cells'],
                'escaped':world['ever_exited'],'stop':world['stopped'],'turns':world['turn'],
                'trace_sha256':receipt['trace_sha256'],'origin':'external synthetic legal-material simulation'})
    for policy in ('model','model_no_goal','recovery_rule','greedy','random_low_stop','random'):
        x=[r for r in rows if r['policy']==policy]
        checks.append({'policy':policy,'episodes':len(x),'escape_rate':sum(r['escaped'] for r in x)/len(x),
            'mean_new':sum(r['new'] for r in x)/len(x),'stop_selected':sum(r['stop'] for r in x),'early_stop':sum(r['stop'] and r['turns']<40 for r in x)})
    report={'schema':'pref-p3-outcome/v1','train_seed_count':len(train_seeds),'test_seed_count':len(test_seeds),
        'train_seed_disjoint':set(train_seeds).isdisjoint(test_seeds),
        'head_sha256':head_sha,'ablation_head_sha256':no_goal_sha,
        'heldout_weak_pair_agreement':{'goal':heldout_agreement(make_training(test_seeds),head['weights']),
          'no_goal':heldout_agreement(make_training(test_seeds),head_no_goal['weights'])},'trained_cpu_steps':head['adam_steps']+head_no_goal['adam_steps'],'weak_pairs':len(pairs),
        'axes':'verified trap escape, synthetic legal new ink, STOP; not owner aesthetics',
        'native_receipts':0,'human_votes':0,'new_pretrained_model_forwards':0,
        'artifact_gallery':'144 original external synthetic material PNG+JSON trajectory pairs',
        'outcomes':checks,'rows':rows}
    (out/'p3-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    titles=[str(z) for z in (9000,9001,9002,9003)]
    cols=('model','model_no_goal','recovery_rule','greedy','random_low_stop','random')
    cells=''.join('<tr><th>'+str(seed)+'</th>'+''.join('<td><p>'+policy+'</p><img src="p3-'+str(seed)+'-'+policy+'.png" width="120" height="120"/><p>'+str(next(r['new'] for r in rows if r['seed']==seed and r['policy']==policy))+' new cells</p></td>' for policy in cols)+'</tr>' for seed in (9000,9001,9002,9003))
    html_doc='<html><head><meta charset="utf-8"/><title>P3 original synthetic trap bestiary</title></head><body><h1>P3 external synthetic recovery bestiary</h1><p>Original material-derived PNGs. Red=protected, slate=obstacle, orange=ink; model outputs from small CPU-trained weak-label heads, NOT native Godot, pretrained d1 or human preference.</p><table border="1">'+cells+'</table><p>All 144 individually preserved PNGs and raw legal-action receipts are in this artifact. See p3-report.json for the entire unselected cohort.</p></body></html>\n'
    (out/'P3_bestiary.html').write_text(html_doc)
    (out/'SOURCE_INVENTORY.txt').write_text('\n'.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}' for f in sorted(out.iterdir()))+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    args=ap.parse_args();print(json.dumps(main(args.output),indent=2))
