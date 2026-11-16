"""P8: genuinely Adam-trained action-order temporal encoder for population breeding.

TRAIN-only temporal convolution is fitted to distinguish ordered host-legal action
trajectories from multiset-preserving reversal and scrambling. Frozen encoder
cells compete with raw MAP, P7 PCA MAP, random ancestry and reward PBT.
All episodes are external synthetic approximations; NO native CyberSand, SLM,
self-play, policy gradient or game modifications are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .population_lab import NICHES, DIRECTIONS, episode, evaluate, draw_svg, stable_seed
    from .population_p1 import candidate_behaviour, load_policy, model_sha
    from .population_p3 import long_horizon
    from .population_p5 import challenge_episode, challenge_eval
    from .population_p6 import split_seeds, train_child, topology_bin
    from .population_p7 import (fit_encoder, learned_cell, phenotype, archive_add)
except ImportError:
    from population_lab import NICHES, DIRECTIONS, episode, evaluate, draw_svg, stable_seed
    from population_p1 import candidate_behaviour, load_policy, model_sha
    from population_p3 import long_horizon
    from population_p5 import challenge_episode, challenge_eval
    from population_p6 import split_seeds, train_child, topology_bin
    from population_p7 import fit_encoder, learned_cell, phenotype, archive_add

ACTIONS = tuple(DIRECTIONS) + ("RETRACE", "STOP")
ARMS = ("reward_pbt", "raw_map", "pca_map", "temporal_map", "random_parents")
SEQ_LEN = 30
KERNEL = 5
HIDDEN = 18
DIM = 3


def sequence(world, length=SEQ_LEN):
    """Pure ordered action tokens; no spatial occupancy or hand-coded style features."""
    a = np.zeros((length, len(ACTIONS)), dtype=np.float64)
    if len(world.history) > length:
        raise ValueError("episode exceeds fixed encoder horizon")
    for i, item in enumerate(world.history):
        a[i, ACTIONS.index(item["action"])] = 1
    return a, len(world.history)


def disorder(tokens, n, method, rng):
    """Controls preserve the action histogram *exactly*, including STOP/RETRACE."""
    out = tokens.copy()
    if method == "reverse":
        out[:n] = out[:n][::-1]
    elif method == "shuffle":
        if n > 1:
            out[:n] = out[:n][rng.permutation(n)]
    else:
        raise ValueError(method)
    if not np.array_equal(out[:n].sum(axis=0), tokens[:n].sum(axis=0)):
        raise AssertionError("temporal negative changed action counts")
    return out


def windows(tokens):
    """Explicit five-step temporal receptive field, with zero edge padding."""
    if tokens.ndim != 3 or tokens.shape[1:] != (SEQ_LEN, len(ACTIONS)):
        raise ValueError("invalid temporal batch")
    padded = np.pad(tokens, ((0, 0), (KERNEL//2, KERNEL//2), (0, 0)))
    return np.concatenate([padded[:, j:j+SEQ_LEN, :] for j in range(KERNEL)], axis=-1)


def init_parameters(seed):
    r = np.random.default_rng(seed)
    return dict(W1=r.normal(0, .16, (len(ACTIONS)*KERNEL, HIDDEN)), b1=np.zeros(HIDDEN),
                W2=r.normal(0, .28, (HIDDEN, DIM)), b2=np.zeros(DIM),
                W3=r.normal(0, .4, (DIM, 3)), b3=np.zeros(3))


def forward(x, lengths, p):
    w = windows(x)
    m = (np.arange(SEQ_LEN)[None, :] < lengths[:, None]).astype(float)
    hidden = np.tanh(w @ p["W1"] + p["b1"])
    pooled = (hidden*m[...,None]).sum(axis=1) / np.maximum(1, lengths)[:, None]
    z = np.tanh(pooled @ p["W2"] + p["b2"])
    logits = z @ p["W3"] + p["b3"]
    return logits, z, (w, m, hidden, pooled)


def gradient_step(x, lengths, labels, p, m1, m2, step, lr):
    logits, z, (w, m, h, pooled) = forward(x, lengths, p)
    exp = np.exp(np.clip(logits-logits.max(axis=1, keepdims=True), -40, 0))
    prob = exp/exp.sum(axis=1, keepdims=True)
    loss = -np.log(np.maximum(1e-12, prob[np.arange(len(labels)),labels])).mean()
    dlogit = prob.copy()
    dlogit[np.arange(len(labels)), labels] -= 1.
    dlogit /= len(labels)
    grad = {}
    grad["W3"] = z.T @ dlogit
    grad["b3"] = dlogit.sum(axis=0)
    dz = (dlogit @ p["W3"].T)*(1-z*z)
    grad["W2"] = pooled.T @ dz
    grad["b2"] = dz.sum(axis=0)
    dh = (dz @ p["W2"].T)[:,None,:] * (m/np.maximum(1,lengths)[:,None])[:,:,None]*(1-h*h)
    grad["W1"] = np.einsum("btf,bth->fh",w,dh)
    grad["b1"] = dh.sum(axis=(0,1))
    norm = np.sqrt(sum(np.square(g).sum() for g in grad.values()))
    if norm > 5:
        grad = {k:v*5/norm for k,v in grad.items()}
    for key in p:
        m1[key] = .9*m1[key] + .1*grad[key]
        m2[key] = .999*m2[key] + .001*grad[key]*grad[key]
        p[key] -= lr * (m1[key]/(1-.9**step)) / (np.sqrt(m2[key]/(1-.999**step))+1e-8)
    return float(loss)


def labelled_sequences(parents, seed, phase, probe_count, probe_worlds):
    rng = np.random.default_rng(stable_seed("p8",seed,phase,"reordering"))
    world_seeds = [stable_seed("p8",seed,phase,"world",i) for i in range(probe_worlds)]
    samples, lengths, labels, positive = [], [], [], []
    for niche in NICHES:
        for j in range(probe_count+1):
            policy = parents[niche] if j==0 else parents[niche].mutate(
                stable_seed("p8",seed,phase,niche,"probe",j))
            for s in world_seeds:
                # Probe diversity includes protected regions, goal switching and retrace.
                _, world = (episode(policy,niche,s) if j%2 == 0
                            else challenge_episode(policy,niche,s,SEQ_LEN))
                tokens, n = sequence(world)
                if n < 3:   # Very short STOP-only traces carry no temporal signal.
                    continue
                positive.append((tokens,n))
                for label, method in enumerate((None, "reverse", "shuffle")):
                    seq = tokens if method is None else disorder(tokens,n,method,rng)
                    samples.append(seq)
                    lengths.append(n)
                    labels.append(label)
    if len(positive) < 6:
        raise AssertionError("training probe corpus collapsed")
    return np.stack(samples), np.array(lengths,dtype=int), np.array(labels,dtype=int), positive, world_seeds


def encode(tokens, length, enc):
    p = {k:np.asarray(v,dtype=np.float64) for k,v in enc["parameters"].items()}
    _, z, _ = forward(tokens[None], np.asarray([length]),p)
    return z[0]


def temporal_cell(positive, enc):
    """Average per-policy embeddings on SELECTION worlds, quartiles TRAIN-frozen."""
    vec = np.mean([encode(x,n,enc) for x,n in positive],axis=0)
    cuts = np.asarray(enc["cuts"])
    return [int(np.searchsorted(cuts[:,i],vec[i],side="right")) for i in range(DIM)], vec.tolist()


def fit_temporal(parents, seed, probe_count=7, probe_worlds=4, steps=260, batch_size=36):
    if not (4<=probe_count<=40 and 3<=probe_worlds<=20 and 40<=steps<=2000 and 12<=batch_size<=128):
        raise ValueError("unbounded temporal encoder training")
    train = labelled_sequences(parents,seed,"encoder_train",probe_count,probe_worlds)
    valid = labelled_sequences(parents,seed,"encoder_validation",max(4,probe_count//2),3)
    x,n,y,positive,train_seeds = train
    xv,nv,yv,_,val_seeds = valid
    params=init_parameters(stable_seed("p8",seed,"temporal_network"))
    m={k:np.zeros_like(v) for k,v in params.items()}
    v={k:np.zeros_like(v) for k,v in params.items()}
    rng=np.random.default_rng(stable_seed("p8",seed,"encoder_batches"))
    losses=[]
    for i in range(1,steps+1):
        ids=rng.integers(0,len(x),size=batch_size)
        loss=gradient_step(x[ids],n[ids],y[ids],params,m,v,i,.012)
        if i==1 or i%20==0 or i==steps:
            losses.append([i,round(loss,6)])
    val_logits,_,_=forward(xv,nv,params)
    train_logits,_,_=forward(x,n,params)
    positives = np.stack([encode_(tok,l,params) for tok,l in positive])
    cuts = np.quantile(positives,[.25,.5,.75],axis=0)
    if not np.isfinite(cuts).all() or len(np.unique(positives,axis=0))<4:
        raise AssertionError("collapsed temporal train embeddings")
    enc=dict(method="TRAIN-only 5-action-window neural temporal convolution; 3-class original/reverse/shuffle cross entropy",
             seed=seed, parameters={k:val.tolist() for k,val in params.items()},
             cuts=cuts.tolist(), train_world_seeds=train_seeds, validation_world_seeds=val_seeds,
             samples=len(x), optimizer_steps=steps, optimizer="NumPy Adam", loss_history=losses,
             train_accuracy=round(float((train_logits.argmax(axis=1)==y).mean()),6),
             encoder_validation_accuracy=round(float((val_logits.argmax(axis=1)==yv).mean()),6),
             positive_variances=positives.var(axis=0).tolist(),
             limitations=["Temporal order discrimination is a proxy task, not artistic preference",
                          "Training on original vs reversed/shuffled action sequences may exploit trivial order statistics",
                          "Action-only convolution cannot perceive terrain/materials directly",
                          "Only train/validation episodes are used to learn representation and quantile edges"])
    enc["sha256"] = hashlib.sha256(json.dumps(enc,sort_keys=True).encode()).hexdigest()
    return enc


def encode_(tokens, length, params):
    return forward(tokens[None],np.asarray([length]),params)[1][0]


def behaviour_sequences(policy, niche, seeds):
    return [sequence(episode(policy,niche,s)[1]) for s in seeds]


def parent_for(arm,g,child,seed,niche,archive,best,pool):
    if g==0:
        return None
    if arm=="reward_pbt":
        return best
    rng=np.random.default_rng(stable_seed("p8",seed,niche,arm,g,child,"parent"))
    if arm in ("raw_map","pca_map","temporal_map"):
        keys=sorted(archive)
        return archive[keys[int(rng.integers(len(keys)))]]
    return pool[int(rng.integers(len(pool)))]


def select_ids(records,archive,arm,k,seed):
    ranked=sorted(records,key=lambda r:(-r["selection"]["mean_reward"],r["id"]))
    if arm in ("raw_map","pca_map","temporal_map"):
        selected=[r["id"] for r in sorted(archive.values(),key=lambda r:(-r["selection"]["mean_reward"],r["id"]))[:k]]
        selected += [r["id"] for r in ranked if r["id"] not in selected][:k-len(selected)]
        return selected
    if arm=="reward_pbt":
        return [r["id"] for r in ranked[:k]]
    rng=np.random.default_rng(seed)
    return [records[int(i)]["id"] for i in rng.choice(len(records),size=k,replace=False)]


def run_p8(p0_output,output,seed=638,generations=3,offspring=4,train_episodes=8,
           selection_episodes=5,holdout_episodes=6,horizon=60,portfolio=4,
           encoder_steps=260,probe_count=7,probe_worlds=4):
    if not(2<=generations<=6 and 3<=offspring<=10 and 4<=train_episodes<=40 and
           3<=selection_episodes<=24 and 3<=holdout_episodes<=24 and 31<=horizon<=128 and
           2<=portfolio<=offspring):
        raise ValueError("P8 unbounded campaign")
    p0_output,output=Path(p0_output),Path(output)
    source=json.loads((p0_output/"results.json").read_text())
    if source.get("format")!="population-research-v1" or source.get("language_model") is not False:
        raise ValueError("P8 requires verified original synthetic P0 provenance")
    splits=split_seeds(seed,train_episodes,selection_episodes,holdout_episodes,generations)
    parents={n:load_policy(p0_output/"models"/(n+".json")) for n in NICHES}
    temporal=fit_temporal(parents,seed,probe_count,probe_worlds,encoder_steps)
    pca=fit_encoder(parents,seed,probe_count=max(4,probe_count),probe_worlds=probe_worlds)
    p0_seeds={stable_seed(source["seed"],part,i) for part,count in
              (("train",source["train_seed_count"]),("selection",source["selection_seed_count"]),
               ("test",source["heldout_seed_count"])) for i in range(count)}
    all_seeds=(sum(splits["train"],[])+splits["selection"]+splits["holdout"]+
               temporal["train_world_seeds"]+temporal["validation_world_seeds"]+pca["train_seeds"])
    if len(all_seeds)!=len(set(all_seeds)) or set(all_seeds)&p0_seeds:
        raise AssertionError("P0/P8 fit/selection/holdout seed leakage")
    output.mkdir(parents=True,exist_ok=False)
    (output/"models").mkdir()
    (output/"gallery").mkdir()
    (output/"temporal_encoder.json").write_text(json.dumps(temporal,indent=2)+"\n")
    (output/"pca_encoder.json").write_text(json.dumps(pca,indent=2)+"\n")
    records, comparisons, models=[],{},{}
    for niche in NICHES:
        comparisons[niche]={}
        for arm in ARMS:
            archive, members, pool, best={},[],[],None
            occupancy=[]
            for g in range(generations):
                batch=[]
                for i in range(offspring):
                    par=parent_for(arm,g,i,seed,niche,archive,best,pool)
                    parent=parents[niche] if par is None else models[par["id"]]
                    ms=stable_seed("p8",seed,niche,g,i,"mutation")
                    policy=parent.mutate(ms)
                    ident=f"{niche}-{arm}-g{g}-{i}"
                    start_sha=model_sha(policy)
                    training=train_child(policy,niche,splits["train"][g])
                    selection=candidate_behaviour(policy,niche,splits["selection"])
                    raw_cell=list(topology_bin(selection))
                    pca_cell,_=learned_cell(phenotype(policy,niche,splits["selection"]),pca)
                    seqs=behaviour_sequences(policy,niche,splits["selection"])
                    temp_cell,z=temporal_cell(seqs,temporal)
                    rec=dict(id=ident,niche=niche,arm=arm,generation=g,offspring=i,
                             parent=None if par is None else par["id"],mutation_seed=ms,
                             initial_sha256=start_sha,final_sha256=model_sha(policy),
                             training=training,selection=selection,raw_cell=raw_cell,
                             pca_cell=pca_cell,temporal_cell=temp_cell,
                             temporal_embedding=np.round(z,6).tolist())
                    if rec["initial_sha256"]==rec["final_sha256"]:
                        raise AssertionError("no actual optimizer update")
                    batch.append(rec)
                    members.append(rec)
                    records.append(rec)
                    models[ident]=policy
                    (output/"models"/(ident+".json")).write_text(json.dumps(policy.serialized(),sort_keys=True)+"\n")
                if arm in ("raw_map","pca_map","temporal_map"):
                    archive=archive_add(archive,batch,{"raw_map":"raw_cell","pca_map":"pca_cell",
                                                      "temporal_map":"temporal_cell"}[arm])
                best=max(members,key=lambda r:(r["selection"]["mean_reward"],r["id"]))
                rng=np.random.default_rng(stable_seed("p8",seed,niche,arm,g,"pool"))
                pool=[batch[int(i)] for i in rng.permutation(len(batch))]
                occupancy.append(len(archive))
            selected=select_ids(members,archive,arm,portfolio,
                                stable_seed("p8",seed,niche,arm,"portfolio"))
            chosen=[models[x] for x in selected]
            comp=dict(champion_id=best["id"], selected_ids=selected, candidate_count=len(members),
                      selection_archive_cells=len(archive),occupancy_by_generation=occupancy,
                      selection_raw_cells=len({tuple(r["raw_cell"]) for r in members}),
                      selection_pca_cells=len({tuple(r["pca_cell"]) for r in members}),
                      selection_temporal_cells=len({tuple(r["temporal_cell"]) for r in members}))
            # No holdout world is opened until final portfolio has been frozen.
            comp["heldout_long"]=long_horizon(chosen,niche,splits["holdout"],horizon)
            comp["heldout_reward"]=evaluate(models[best["id"]],niche,splits["holdout"])
            comp["challenge"]=challenge_eval(chosen,niche,splits["holdout"],horizon)
            holdemb=[]
            for p in chosen:
                sequence_embeddings=[]
                for s in splits["holdout"]:
                    _,w=challenge_episode(p,niche,s,horizon)
                    # Representation fixed at first 30 actions to match encoder training.
                    trunc=type("TruncatedHistory",(),{"history":w.history[:SEQ_LEN]})()
                    t,l=sequence(trunc)
                    sequence_embeddings.append(encode(t,l,temporal))
                holdemb.append(np.mean(sequence_embeddings,axis=0))
            comp["heldout_temporal_distance"]=round(float(np.mean([
                np.linalg.norm(holdemb[i]-holdemb[j]) for i in range(portfolio)
                for j in range(i+1,portfolio)])),6)
            comparisons[niche][arm]=comp
            for j,ident in enumerate(selected):
                _,w=challenge_episode(models[ident],niche,splits["holdout"][0],horizon)
                path=f"{niche}-{arm}-{j}"
                (output/"gallery"/(path+".svg")).write_text(draw_svg(w))
                (output/"gallery"/(path+"-trace.json")).write_text(json.dumps(dict(
                    synthetic=True,native=False,slm=False,seed=splits["holdout"][0],policy_id=ident,
                    trajectory=w.history,painted=sorted(w.painted),obstacles=sorted(w.obstacles),
                    protected=sorted(w.protected)),indent=2)+"\n")
    summary={arm:dict(
        heldout_diversity=round(float(np.mean([comparisons[n][arm]["heldout_long"]["mean_distance"]
                                                for n in NICHES])),6),
        heldout_reward=round(float(np.mean([comparisons[n][arm]["heldout_reward"]["mean_reward"]
                                             for n in NICHES])),6),
        challenge_retraces=round(float(np.mean([comparisons[n][arm]["challenge"]["mean_retraces"]
                                                for n in NICHES])),6),
        heldout_temporal_distance=round(float(np.mean([comparisons[n][arm]["heldout_temporal_distance"]
                                                     for n in NICHES])),6),
        selection_archive_cells=sum(comparisons[n][arm]["selection_archive_cells"] for n in NICHES))
        for arm in ARMS}
    result=dict(format="population-research-p8",kind="TRAIN_ONLY_NEURAL_TEMPORAL_MAP_ABLATION",
                synthetic=True,native=False,language_model=False,reinforcement_learning=False,
                self_play=False,game_code_modified=False,seed=seed,arms=list(ARMS),
                candidate_count=len(records),adam_policy_updates=sum(r["training"]["adam_steps"] for r in records),
                adam_encoder_updates=encoder_steps, encoder_sha256=temporal["sha256"],
                pca_sha256=pca["sha256"],generations=generations,offspring=offspring,
                portfolio=portfolio,horizon=horizon,records=records,comparison=comparisons,
                summary=summary,splits_sha256={k:hashlib.sha256(json.dumps(v).encode()).hexdigest()
                                              for k,v in splits.items()},
                limitations=["External synthetic grid worlds only, no native CyberSand gameplay",
                             "Policies are actual trained tiny MLPs, not language models/SLM students",
                             "Temporal representation is learned for ordered-vs-permuted classification, not aesthetic quality",
                             "It may exploit elementary action-transition statistics rather than meaningful planning",
                             "Five evolutionary treatments have different ancestry after generation 0",
                             "Training is teacher-augmented supervised Adam, not self-play or policy-gradient RL",
                             "A fixed selection-world set is reused between generations; overfitting possible",
                             "All actions legal because the host masks choices, not because models learned safety",
                             "Two cohorts cannot establish population-level statistical superiority"])
    (output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--p0-output",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--seed",type=int,default=638)
    a.add_argument("--generations",type=int,default=3)
    a.add_argument("--offspring",type=int,default=4)
    a.add_argument("--train-episodes",type=int,default=8)
    a.add_argument("--selection-episodes",type=int,default=5)
    a.add_argument("--holdout-episodes",type=int,default=6)
    a.add_argument("--horizon",type=int,default=60)
    a.add_argument("--portfolio",type=int,default=4)
    a.add_argument("--encoder-steps",type=int,default=260)
    a.add_argument("--probe-count",type=int,default=7)
    a.add_argument("--probe-worlds",type=int,default=4)
    ns=a.parse_args()
    out=run_p8(ns.p0_output,ns.output,ns.seed,ns.generations,ns.offspring,
               ns.train_episodes,ns.selection_episodes,ns.holdout_episodes,ns.horizon,
               ns.portfolio,ns.encoder_steps,ns.probe_count,ns.probe_worlds)
    print(json.dumps(dict(seed=ns.seed,candidates=out["candidate_count"],
                          policy_adam_updates=out["adam_policy_updates"],
                          encoder_adam_updates=out["adam_encoder_updates"],
                          encoder_validation_accuracy=json.loads((ns.output/"temporal_encoder.json").read_text())["encoder_validation_accuracy"],
                          summary=out["summary"]),indent=2))

if __name__=="__main__":
    main()
