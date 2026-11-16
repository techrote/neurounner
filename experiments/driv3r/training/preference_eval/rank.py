"""Source-fenced, genuinely trained geometric PROXY ranker. No human ratings or gameplay changes.

The dataset contains model-forward synthetic W0 and model-free controls, never
native Godot outcomes. The pairwise targets are declared geometric formulas,
not aesthetically judged preference votes. No additional neural worm runs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from itertools import combinations
from pathlib import Path

DATA_SHA = '2c9404a658ac11f2ec90cc6eefd8f61bc95d9dc7fe9e5506f1a7b429c47dc879'
FEATURES = ('cells','xspan','yspan','edges','junctions','cycles','endpoints',
            'obstacle_contacts','diagonal_contacts','western_cells')


def validate(path):
    blob=Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=DATA_SHA:
        raise ValueError('dataset changed without a new independently audited source manifest')
    rows=list(csv.DictReader(blob.decode('utf-8').splitlines()))
    if len(rows)!=32:raise ValueError('missing original image rows')
    source={'lfm_w13':{'source_sha':'245290cab2352381dbd1eaf16374c577138b4ade','run_id':43771476464,
        'original_archive_sha256':'0e69b23c100a00b438d4e2d9d07fa4550e276368e0ad4635e38a6c858f79b95e'},
        'sprocket_f3':{'source_sha':'e56e0aac49b310cc1e00aafe3ffeddb90eca6e38','run_id':43769025429,
        'original_archive_sha256':'5eb64e5f7d5335809f631aa11caf9edbfcf1a67f546601d5d14e795cc61f7af6'}}
    items=[]
    for r in rows:
        f=r['family']; seed=int(r['seed']);role=r['controller']
        if f not in source or role not in ('learned','random','wall','fidget'):
            raise ValueError('unrecognized source or controller')
        z={'id':f'{f}-{seed}-{role}','model_family':f,'world_seed':seed,'controller':role,
           'decisions':128,'evidence_class':'trained-model-forward/synthetic-W0' if role=='learned' else 'model-free/synthetic-W0',
           'original_png_sha256':r['png_sha256'],**source[f],
           'features':{k:int(r[k]) for k in FEATURES+('components',)}}
        if f=='lfm_w13' and seed in (6143,6167):
            z['original_archive_sha256']='36fa8876f249b0263d19dc40bea17731e8a4776da9cc82400189a876980380ff'
        items.append(z)
    if len({x['id'] for x in items})!=32: raise ValueError('duplicated original image IDs')
    worlds={}
    for x in items:
        if x['decisions']!=128 or len(x['original_png_sha256'])!=64 or x['features']['components']<1 or x['features']['cells']<1:
            raise ValueError('source identity or geometry invalid')
        worlds.setdefault((x['model_family'],x['world_seed']),[]).append(x)
    if len(worlds)!=8 or any({x['controller'] for x in v}!={'learned','random','wall','fidget'} for v in worlds.values()):
        raise ValueError('world comparison not aligned')
    return worlds


def proxy(x,axis):
    f=x['features']; n=float(f['cells']); footprint=float(f['xspan']*f['yspan'])
    if axis=='spatial_extent':
        return 0.55*math.log1p(n)/math.log1p(129.) + 0.45*math.log1p(footprint)/math.log1p(96*96)
    if axis=='local_weave':
        return 0.40*(f['cycles']/n) + 0.35*(f['junctions']/n) + 0.25*(n/footprint)
    raise ValueError('unsupported axis')


def create_pairs(worlds,axis):
    pairs=[];ties=[]
    for world,items in sorted(worlds.items()):
        for a,b in combinations(sorted(items,key=lambda x:x['controller']),2):
            margin=proxy(a,axis)-proxy(b,axis)
            if abs(margin)<0.002:
                ties.append((world,a['id'],b['id'],margin));continue
            pairs.append({'world':f'{world[0]}:{world[1]}','a':a,'b':b,'label':int(margin>0),
                          'proxy_margin':round(margin,8),'axis':axis})
    return pairs,ties


def featurize(item):
    f=item['features'];n=f['cells'];area=max(1,f['xspan']*f['yspan'])
    return [math.log1p(n),math.log1p(area),f['cycles']/n,f['junctions']/n,
            n/area,math.log1p(f['endpoints']),math.log1p(f['obstacle_contacts']),
            math.log1p(f['xspan']),math.log1p(f['yspan']),f['western_cells']/n]


def standardizer(pairs):
    xs=[featurize(item) for p in pairs for item in (p['a'],p['b'])]
    mean=[sum(x[j] for x in xs)/len(xs) for j in range(len(xs[0]))]
    std=[max(1e-6,(sum((x[j]-mean[j])**2 for x in xs)/len(xs))**.5) for j in range(len(mean))]
    return mean,std


def differences(pairs,mean,std):
    return [([(a-b)/std[j] for j,(a,b) in enumerate(zip(featurize(p['a']),featurize(p['b'])))],p['label']) for p in pairs]


def sigmoid(z):
    return 1/(1+math.exp(-max(-35.,min(35.,z))))


def fit(pairs,epochs=450,lr=0.055,reg=0.05):
    mean,std=standardizer(pairs)
    xy=differences(pairs,mean,std)
    weights=[0.]*len(mean);m=[0.]*len(mean);v=[0.]*len(mean)
    history=[];step=0
    for e in range(epochs):
        grad=[0.]*len(weights)
        loss=0.
        for x,y in xy:
            z=sum(a*b for a,b in zip(x,weights))
            p=sigmoid(z);delta=p-y
            loss+=-(y*math.log(max(1e-12,p))+(1-y)*math.log(max(1e-12,1-p)))
            for j in range(len(weights)):grad[j]+=delta*x[j]
        loss/=len(xy)
        for j in range(len(weights)):
            grad[j]=grad[j]/len(xy)+reg*weights[j]
        step+=1
        for j,g in enumerate(grad):
            m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
            weights[j]-=lr*(m[j]/(1-.9**step))/(math.sqrt(v[j]/(1-.999**step))+1e-8)
        if e in (0,9,49,149,epochs-1):history.append([e+1,round(loss,6)])
    return {'weights':weights,'std':std,'mean':mean,'epochs':epochs,'learning_rate':lr,'l2':reg,'train_loss_curve':history}


def evaluate(model,pairs):
    if not pairs:return {'pairs':0,'accuracy':None}
    data=differences(pairs,model['mean'],model['std'])
    probs=[sigmoid(sum(x*w for x,w in zip(row,model['weights']))) for row,y in data]
    ok=sum((p>=.5)==bool(y) for p,(row,y) in zip(probs,data))
    brier=sum((p-y)**2 for p,(row,y) in zip(probs,data))/len(probs)
    return {'pairs':len(pairs),'correct':ok,'accuracy':round(ok/len(pairs),4),'brier':round(brier,4)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(Path(__file__).with_name('source_features.csv')))
    ap.add_argument('--output-dir',default='evidence/preference');args=ap.parse_args()
    worlds=validate(args.data)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    report={'schema':'cybersand-preference-proxy-result/v1','dataset_sha256':DATA_SHA,
            'evidence':'32 source-cited original PNGs; 8 model-forward synthetic W0 and 24 matched model-free W0 controls',
            'real_model_inference_executed_here':False,'training_type':'actual stdlib Adam pairwise logistic proxy-head optimization',
            'human_owner_votes':0,'aesthetic_ground_truth':False,'native_replay':False,'axes':{}}
    pair_export=[]
    for axis in ('spatial_extent','local_weave'):
        pairs,ties=create_pairs(worlds,axis)
        ids=sorted({p['world'] for p in pairs})
        folds=[ids[k::4] for k in range(4)]
        folds_report=[]
        for holdout in folds:
            train=[p for p in pairs if p['world'] not in holdout]
            test=[p for p in pairs if p['world'] in holdout]
            mod=fit(train);met=evaluate(mod,test)
            folds_report.append({'held_out_worlds':holdout,'train_pairs':len(train),**met})
        full=fit(pairs)
        train_met=evaluate(full,pairs)
        baselines={}
        for factor in ('cells','xspan'):
            pred=[int(p['a']['features'][factor]>p['b']['features'][factor]) for p in pairs]
            baselines[factor+'_only_correct']=sum(y==p['label'] for y,p in zip(pred,pairs))
        report['axes'][axis]={'pairs':len(pairs),'abstained_ties':len(ties),
            'train':train_met,'heldout_folds':folds_report,
            'cv_accuracy':round(sum(z['correct'] for z in folds_report)/sum(z['pairs'] for z in folds_report),4),
            'cv_brier_unweighted':round(sum(z['brier'] for z in folds_report)/len(folds_report),4),
            'baselines':baselines,'train_loss_curve':full['train_loss_curve']}
        (out/(axis+'-local-checkpoint.json')).write_text(json.dumps(full,indent=2,sort_keys=True)+'\n')
        for p in pairs:
            pair_export.append({'axis':axis,'world':p['world'],'a_png_sha256':p['a']['original_png_sha256'],
             'b_png_sha256':p['b']['original_png_sha256'],'a':p['a']['id'],'b':p['b']['id'],
             'label':p['label'],'label_origin':'algorithmic geometric proxy; not a human vote',
             'margin':p['proxy_margin']})
    (out/'pairwise_proxy.json').write_text(json.dumps(pair_export,indent=2,sort_keys=True)+'\n')
    (out/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    # Separate P2: predict future observed W0 events from first-64 host receipts.
    # Source-frozen, model-side tiny CPU head; never a human aesthetic reward.
    import trajectory_p2
    trajectory_p2.run(out)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
