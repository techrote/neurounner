#!/usr/bin/env python3
"""W18: separate synthetic material/protection-aware host; Sprocket auxiliary policy.

Original W17 head is a frozen or late-layer fine-tuned auxiliary parent, NOT
Sprocket-500M weights. Source-pinned W0 owns geometric movement. This external
research wrapper owns an additional MATERIAL/PROTECTED legal mask; not native.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, random, resource, struct, time, zlib
from collections import Counter
from pathlib import Path

VERSION = 'sprocket-w18-material-overlay-residual-v1'
ROOT = 'kandivault/sprocket-500m'
REVISION = 'eb157d4ddddc4619e661d4a2426bb2dd000fc668'
EOS = 5
W17_SHA = '3d0fe44f303851db6f34f489eb17ff2b1f6dc5a5cfe569b0637716306594aec3'
W0_SHA = '81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e'
ACTIONS = ('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
FAMILIES = ('greebles','pipework','architecture')
MATERIALS = ('sand','stone','water','metal','protected','obstacle')
DIRECTIONS = ((1,0),(0,1),(-1,0),(0,-1))
TRAIN = (401,409,419,421)
HELDOUT = (2801,2819,2833)
WAVES = 160
STEPS = 128


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def w0_module(path):
    spec=importlib.util.spec_from_file_location('w0_external_pinned',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ACTIONS==ACTIONS and module.FAMILIES==FAMILIES
    return module


def material(seed,x,y):
    # Spatially correlated 7-cell zones with single-cell features. No game physics.
    v=((x//7)*1103515245 ^ (y//7)*2654435761 ^ seed*2246822519) & 0xffffffff
    return MATERIALS[v % 4]


def protected(seed,x,y):
    # W0 starting cell is outside these protected fixtures. Host must not paint them.
    fence=19+(seed%9)
    return ((x==fence and 25<=y<=70 and y%17 not in (0,1,2,3))
            or (x>11 and ((x*11+y*19+seed)%89==0)))


def augment(obs,seed):
    o=dict(obs)
    rows=[]; blocks=[]
    x,y=obs['x'],obs['y']
    for dy in range(-2,3):
        row=[]; block=[]
        for dx in range(-2,3):
            xx,yy=x+dx,y+dy
            geo=obs['local_5x5'][dy+2][dx+2]
            kind=('obstacle' if geo=='#' else 'protected' if protected(seed,xx,yy)
                  else material(seed,xx,yy))
            row.append(kind)
            block.append('#' if geo=='#' or (kind=='protected' and (dx,dy)!=(0,0)) else geo)
        rows.append(row);blocks.append(''.join(block))
    o['material_5x5']=rows
    o['local_5x5']=blocks
    safe=[]
    for a in obs['legal']:
        if a in ('STRAIGHT','LEFT','RIGHT'):
            rot={'STRAIGHT':0,'LEFT':-1,'RIGHT':1}[a]
            dx,dy=DIRECTIONS[(o['heading']+rot)%4]
            if not protected(seed,x+dx,y+dy):safe.append(a)
        elif a=='BRANCH':
            # W0's BRANCH chooses the first ORIGINAL unvisited R,L,S arm.
            # Only retain if that precise implied arm avoids protected cells.
            first=None
            for cand in ('RIGHT','LEFT','STRAIGHT'):
                rot={'STRAIGHT':0,'LEFT':-1,'RIGHT':1}[cand]
                dx,dy=DIRECTIONS[(o['heading']+rot)%4]
                if cand in obs['legal'] and obs['local_5x5'][dy+2][dx+2]=='.':
                    first=(x+dx,y+dy);break
            if first and not protected(seed,*first):safe.append(a)
        else:safe.append(a)
    assert 'STOP' in safe and set(safe).issubset(obs['legal'])
    o['legal']=safe
    return o


def geometric_features(obs):
    symbols=' .#+@'.strip()  # .#+@
    assert symbols=='.#+@'
    g=obs['local_5x5']
    v=[float(c==s) for row in g for c in row for s in symbols]
    v += [float(obs['heading']==i) for i in range(4)]
    v += [float(obs['family']==f) for f in FAMILIES]
    tick=obs['tick']
    v += [math.sin(2*math.pi*tick/p) for p in (6,9,12,13,15,17,32)]
    v += [math.cos(2*math.pi*tick/p) for p in (6,9,12,13,15,17,32)]
    v += [min(tick/128,1.),obs['x']/96,obs['y']/96]
    v += [float(a in obs['legal']) for a in ACTIONS]
    recent=obs['last_actions']
    v += [sum(a==move for a in recent)/4 for move in ACTIONS]
    v += [float(bool(recent) and recent[-1]==move) for move in ACTIONS]
    assert len(v)==142
    return v


def material_features(obs):
    row=[float(c==m) for rr in obs['material_5x5'] for c in rr for m in MATERIALS]
    assert len(row)==150
    return row


def model_class(torch):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base=torch.nn.Sequential(torch.nn.Linear(142,64),torch.nn.Tanh(),
                torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,6))
            self.material=torch.nn.Sequential(torch.nn.Linear(150,32),torch.nn.Tanh(),
                torch.nn.Linear(32,6))
            # Exactly frozen original 142-feature policy before any material updates.
            for p in self.base.parameters():p.requires_grad_(False)
        def forward(self,base,mat):
            return self.base(base)+self.material(mat)
    return Policy


def teacher(obs):
    legal=[a for a in obs['legal'] if a!='STOP' or obs['tick']>=124]
    if not legal:return 'STOP'
    priorities={'greebles':'metal','pipework':'water','architecture':'stone'}
    target=priorities[obs['family']]
    scores={}
    for a in legal:
        if a in ('STRAIGHT','LEFT','RIGHT'):
            rot={'STRAIGHT':0,'LEFT':-1,'RIGHT':1}[a]
            dx,dy=DIRECTIONS[(obs['heading']+rot)%4]
            tile=obs['material_5x5'][2+dy][2+dx]
            fresh=obs['local_5x5'][2+dy][2+dx]=='.'
            scores[a]=2.2*float(tile==target)+1.5*float(fresh)+0.15*float(a=='STRAIGHT')
            if tile=='water' and obs['family']=='architecture':scores[a]-=1.0
            if len(obs['last_actions'])>=3 and len(set(obs['last_actions'][-3:]))==1:
                scores[a]+=0.6*float(a!='STRAIGHT')
        elif a=='BRANCH':
            fresh=sum(obs['local_5x5'][2+DIRECTIONS[(obs['heading']+r)%4][1]]
                      [2+DIRECTIONS[(obs['heading']+r)%4][0]]=='.' for r in (0,-1,1))
            scores[a]=2.5 if fresh>=2 and obs['tick']%13 in (3,4,5) else -2.5
            if 'BRANCH' in obs['last_actions'][-3:]:scores[a]-=2.
        elif a=='RETRACE':
            frontier=sum(obs['local_5x5'][2+DIRECTIONS[(obs['heading']+r)%4][1]]
                      [2+DIRECTIONS[(obs['heading']+r)%4][0]]=='.' for r in (0,-1,1))
            scores[a]=3.4 if frontier==0 else -2.0
        else:scores[a]=-10.0
    return max(legal,key=lambda a:scores[a])


def draw(seed,family,tick):
    s=f'{seed}/{family}/{tick}/w18-host-crn'.encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8],'big')/2**64


def choose(logits,legal,temperature,u):
    if temperature==0:return max(legal,key=lambda a:logits[ACTIONS.index(a)])
    z=[logits[ACTIONS.index(a)]/temperature for a in legal]
    mx=max(z);weights=[math.exp(v-mx) for v in z];total=sum(weights)
    acc=0.
    for a,w in zip(legal,weights):
        acc+=w/total
        if u<acc:return a
    return legal[-1]


def rollout(torch,w0,policy,seed,family,name,temp=0):
    decisions=[];blocked=0
    def select(raw):
        nonlocal blocked
        o=augment(raw,seed)
        blocked+=len(raw['legal'])-len(o['legal'])
        if name=='teacher':a=teacher(o);logits=None
        elif name=='random':
            options=[a for a in o['legal'] if a!='STOP' or o['tick']>=124]
            a=options[min(int(draw(seed,family,o['tick'])*len(options)),len(options)-1)]
            logits=None
        elif name=='wall':a=next((a for a in ('RIGHT','STRAIGHT','LEFT','RETRACE','STOP') if a in o['legal']), 'STOP');logits=None
        else:
            with torch.no_grad():
                logits=policy(torch.tensor([geometric_features(o)],dtype=torch.float32),
                              torch.tensor([material_features(o)],dtype=torch.float32))[0].tolist()
            legal=[a for a in o['legal'] if a!='STOP' or o['tick']>=124]
            a=choose(logits,legal,temp,draw(seed,family,o['tick']))
        assert a in o['legal']
        decisions.append({'tick':o['tick'],'material_5x5':o['material_5x5'],
            'host_legal':o['legal'],'original_w0_legal':raw['legal'],
            'selected':a,'actual_head_forward':logits is not None,'six_logits':logits})
        return a
    result=w0.run(seed,'callback',family,STEPS,selector=select)
    assert len(result['events'])==len(decisions)
    assert all(e['raw_legal'] and e['applied']==d['selected'] for e,d in zip(result['events'],decisions))
    assert all(not protected(seed,*p) for p in result['trajectory'])
    result.update(mode=name,engine='W18-source-pinned-W0-plus-external-synthetic-material-protection',
                  material_decisions=decisions,material_overlay_version=VERSION,
                  additional_host_legal_filters=blocked,
                  original_head_forwards=sum(d['actual_head_forward'] for d in decisions))
    return result


def to_rows(results,seed):
    rows=[]
    for rec in results:
        for e,d in zip(rec['events'],rec['material_decisions']):
            obs=augment(e['observation'],seed)
            rows.append((obs,teacher(obs)))
    return rows


def tensorize(torch,rows):
    x=torch.tensor([geometric_features(o) for o,_ in rows],dtype=torch.float32)
    mat=torch.tensor([material_features(o) for o,_ in rows],dtype=torch.float32)
    y=torch.tensor([ACTIONS.index(a) for _,a in rows],dtype=torch.long)
    mask=torch.tensor([[a in o['legal'] for a in ACTIONS] for o,_ in rows],dtype=torch.bool)
    return x,mat,y,mask


def evaluate(torch,policy,rows):
    if not rows:return {'rows':0}
    x,m,y,legal=tensorize(torch,rows)
    with torch.no_grad():
        pred=policy(x,m).masked_fill(~legal,-1e4).argmax(1)
    j=ACTIONS.index('BRANCH')
    tp=int(((pred==j)&(y==j)).sum());fp=int(((pred==j)&(y!=j)).sum());pos=int((y==j).sum())
    return {'rows':len(rows),'accuracy':round(float((pred==y).float().mean()),6),
            'branch_tp':tp,'branch_fp':fp,'branch_positive':pos,
            'branch_precision':round(tp/max(1,tp+fp),6),'branch_recall':round(tp/max(1,pos),6)}


def checkpoint(torch,policy,opt,rng,out,stage,stats):
    path=out/(stage+'.pt')
    torch.save({'version':VERSION,'stage':stage,'model':policy.state_dict(),
      'optimizer':opt.state_dict(),'rng':rng.getstate(),
      'torch_rng_state':torch.random.get_rng_state(),'metrics':stats,
      'root_updates':0,'root_forwards':0,'EOS':EOS},path)
    return {'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size}


def material_png(w0,rec):
    # Original material-protected evaluation visual: unmodified path + synthetic overlay.
    seed=rec['seed'];n=rec['size'];scale=4
    obstacles={tuple(p) for p in rec['obstacles']}
    path={tuple(p):i for i,p in enumerate(rec['trajectory'])}
    branches={tuple(p) for p in rec['branch_points']}
    palette={'sand':(75,60,39),'stone':(65,76,85),'water':(35,65,112),'metal':(84,76,99)}
    pixels=bytearray()
    for y in range(n):
        row=bytearray()
        for x in range(n):
            p=(x,y)
            c=((242,206,70) if p==tuple(rec['trajectory'][-1]) else
               (239,105,215) if p in branches else
               (27,140,205) if p in path else
               (202,50,56) if protected(seed,x,y) else
               (17,19,24) if p in obstacles else palette[material(seed,x,y)])
            row.extend(c*scale)
        for _ in range(scale):pixels.append(0);pixels.extend(row)
    def chunk(t,b):return struct.pack('>I',len(b))+t+b+struct.pack('>I',zlib.crc32(t+b)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',n*scale,n*scale,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels,9))+chunk(b'IEND',b'')


def main(args):
    assert set(TRAIN).isdisjoint(HELDOUT)
    assert (ROOT,REVISION,EOS)==('kandivault/sprocket-500m','eb157d4ddddc4619e661d4a2426bb2dd000fc668',5)
    if args.smoke_only:
        assert material(10,1,2) in MATERIALS[:4]
        assert len(material_features({'material_5x5':[['sand']*5]*5}))==150
        assert choose([0,1,2,3,4,5],['LEFT','RIGHT'],0,0.2)=='RIGHT'
        print('W18 model-free source preflight PASS; no optimizer or native updates')
        return
    begin=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(181818);torch.set_num_threads(1)
    rng=random.Random(181818)
    if sha(args.w17)!=W17_SHA or sha(args.w0)!=W0_SHA:raise ValueError('original SHA admission failed')
    w0=w0_module(args.w0)
    state=torch.load(args.w17,map_location='cpu',weights_only=True)
    assert state['version']=='sprocket-w17-hard-negative-branch-history-calibration-v1'
    assert state['root_optimizer_steps']==0
    model=model_class(torch)()
    # Transfer exact W17 parameters into base at graph-compatible dimensions.
    model.base.load_state_dict({k.removeprefix('layers.'):v for k,v in state['head'].items()},strict=True)
    model.material[-1].weight.data.zero_();model.material[-1].bias.data.zero_()
    baseline=model_class(torch)();baseline.load_state_dict(model.state_dict())
    save(args.output/'provenance.json',{'version':VERSION,'root':ROOT,'revision':REVISION,'eos':EOS,
      'parent_W17_sha256':W17_SHA,'pinned_W0_sha256':W0_SHA,'source_sha':os.getenv('GITHUB_SHA','local'),
      'run_id':os.getenv('GITHUB_RUN_ID','local'),'attempt':os.getenv('GITHUB_RUN_ATTEMPT','local'),
      'script_sha256':sha(__file__),'root_model_loaded':False,'root_optimizer_steps':0,
      'root_forwards':0,'LoRA_updates':0,'native':False,'sealed_FINAL':False,'draft_hold':True,
      'protected_legal_mask_host_owned':True,'materials_are_synthetic_not_native':True})
    train=[];receipts=[]
    for seed in TRAIN:
        for family in FAMILIES:
            for name,p in [('teacher',None),('random',None),('W17-frozen',baseline)]:
                r=rollout(torch,w0,p,seed,family,name)
                train.extend(to_rows([r],seed))
                receipts.append({'seed':seed,'family':family,'controller':name,
                   'decisions':r['steps_executed'],'blocked_legal_choices':r['additional_host_legal_filters']})
    hold=[]
    for seed in HELDOUT:
        for family in FAMILIES:
            for name,p in [('teacher',None),('random',None),('W17-frozen',baseline)]:
                hold.extend(to_rows([rollout(torch,w0,p,seed,family,name)],seed))
    x,m,y,leg=tensorize(torch,train)
    save(args.output/'datasets.json',{'train_seeds':TRAIN,'heldout_seeds':HELDOUT,
       'train_rows':len(train),'heldout_rows':len(hold),'train_teacher':dict(Counter(a for o,a in train)),
       'heldout_teacher':dict(Counter(a for o,a in hold)),'real_closed_loop_host_receipts':receipts,
       'native':False,'counterfactual_world_reachability_verified':False})
    optimizer=torch.optim.AdamW(model.material.parameters(),lr=0.001,weight_decay=0.005)
    history=[];stages=[]
    stages.append({'name':'W17-frozen-material-blind','evaluation':evaluate(torch,baseline,hold)})
    stage_one=None
    for wave,stage in enumerate(('material-residual-only','material-replay-plus-last-layer')):
        if wave:
            stage_one=model_class(torch)();stage_one.load_state_dict(model.state_dict());stage_one.eval()
            for p in model.base[-1].parameters():p.requires_grad_(True)
            optimizer.add_param_group({'params':list(model.base[-1].parameters()),'lr':0.000075})
            # Real on-policy synthetic host replay from stage-one policy, unseen from stage-one supervision.
            extra=[]
            for seed in TRAIN:
                for family in FAMILIES:
                    rr=rollout(torch,w0,stage_one,seed,family,'W18-stage1-sampled',.7)
                    extra.extend(to_rows([rr],seed))
            replay_x,replay_m,replay_y,replay_mask=tensorize(torch,extra)
            save(args.output/'replay.json',{'rows':len(extra),'action_labels':dict(Counter(a for _,a in extra)),
                 'actual_onpolicy_stage_one_forwards':len(extra),'native':False})
        for step in range(WAVES):
            optimizer.zero_grad(set_to_none=True)
            ix=[rng.randrange(len(train)) for _ in range(96)]
            logits=model(x[ix],m[ix]).masked_fill(~leg[ix],-1e4)
            ce=torch.nn.functional.cross_entropy(logits,y[ix])
            # Retain source W17 behaviour when there is no material information.
            if wave:
                blank=torch.zeros_like(m[ix])
                with torch.no_grad():frozen=baseline(x[ix],blank)
                preserve=torch.nn.functional.mse_loss(model(x[ix],blank),frozen)
                rid=[rng.randrange(len(extra)) for _ in range(48)]
                replay_ce=torch.nn.functional.cross_entropy(model(replay_x[rid],replay_m[rid]).masked_fill(~replay_mask[rid],-1e4),replay_y[rid])
                loss=.55*ce+.3*replay_ce+.15*preserve
            else:loss=ce
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),3.0)
            if not bool(torch.isfinite(grad)) or grad.item()<=0:raise ValueError('zero/nonfinite gradient')
            optimizer.step()
            if (step+1)%20==0:
                history.append({'stage':stage,'step':step+1,'total_updates':wave*WAVES+step+1,
                   'loss':round(float(loss.detach()),6),'gradient_norm':round(float(grad),6)})
                save(args.output/'partial-history.json',history)
        stats=evaluate(torch,model,hold)
        cp=checkpoint(torch,model,optimizer,rng,args.output,stage,stats)
        stages.append({'name':stage,'real_adamw_updates':WAVES,'eval':stats,'checkpoint':cp})
        save(args.output/'stages.json',stages)
    outputs=[];gallery=args.output/'original-material-png-json'
    gallery.mkdir(exist_ok=True)
    for seed in HELDOUT:
        for family in FAMILIES:
            for name,policy,temp in [('W17-frozen',baseline,0),('W18-stage1',stage_one,0),
                 ('W18-final',model,0),('W18-final-t07',model,.7),
                 ('random',None,0),('wall',None,0),('teacher',None,0)]:
                rec=rollout(torch,w0,policy,seed,family,name,temp)
                filename=f'{family}-{name}-{seed}'
                path=gallery/(filename+'.json');save(path,rec)
                img=material_png(w0,rec)
                (gallery/(filename+'.png')).write_bytes(img)
                outputs.append({'seed':seed,'family':family,'controller':name,'steps':rec['steps_executed'],
                   'metrics':rec['metrics'],'filtered_actions':rec['additional_host_legal_filters'],
                   'real_head_forwards':rec['original_head_forwards'],
                   'json_sha256':sha(path),'png_sha256':sha(gallery/(filename+'.png')),
                   'trajectory_sha256':hashlib.sha256(json.dumps(rec['trajectory'],separators=(',',':')).encode()).hexdigest()})
        save(args.output/'partial-rollouts.json',outputs)
    assert len(outputs)==63 and all(v['metrics']['invalid_raw_choices']==0 for v in outputs)
    save(args.output/'summary.json',{'status':'complete','version':VERSION,
      'real_aux_adamw_steps':2*WAVES,'root_optimizer_steps':0,'root_forwards':0,'LoRA_updates':0,
      'original_png_json_pairs':len(outputs),'real_head_forwards':sum(v['real_head_forwards'] for v in outputs),
      'stages':stages,'results':outputs,'wall_seconds':time.monotonic()-begin,
      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'synthetic_not_native':True,'protected_violations':0,'materials_not_native':True})
    print(json.dumps({'W18':'complete','genuine_steps':2*WAVES,'original_pngs':len(outputs)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for item in ('w17','w0','output'):p.add_argument('--'+item,type=Path)
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    try:
        if not a.smoke_only and not all((a.w17,a.w0,a.output)):p.error('required inputs missing')
        main(a)
    except BaseException as e:
        if a.output and a.output.exists():save(a.output/'failure.json',{'type':type(e).__name__,'error':str(e),'status':'failed'})
        raise
