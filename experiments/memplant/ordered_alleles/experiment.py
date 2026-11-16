"""DG005: trained, noncommuting game-law edit receiver from real Qwen state.

The donor is a supervised two-scalar DG002 readout, NOT a heterogeneous
native KV transfer. Edit dynamics learn from atomic labels, never sequences.
The action projector is deterministic and public, NOT another trained head.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.memplant.game_state_lab.game_policy import (
    SOURCE_SHA, donor_scores, gate, laws, scenes, target,
)

SEED = 26011605
EDITS = ("H", "V")
SEQUENCES = ("HV", "VH", "HHV", "VVH", "HVH", "VHV")
GAME_OPS = (3, 6, 9, 12)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def edit_bits(hv, op):
    h, v = hv
    if op == "H":
        return 1-h, v ^ h
    if op == "V":
        return h ^ v, 1-v
    raise ValueError("Unknown edit operation")


def compose_bits(hv, sequence):
    for op in sequence:
        hv = edit_bits(hv, op)
    return hv


class SourceDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 2))

    def forward(self, source):
        return self.net(source)


class ContextEdit(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,48), nn.Tanh(),
                                 nn.Linear(48,48), nn.Tanh(), nn.Linear(48,2))

    def forward(self, state_logits, edit):
        return self.net(torch.cat((torch.sigmoid(state_logits),
                                   F.one_hot(edit,2).float()), dim=-1))


class AdditiveEdit(nn.Module):
    """A genuinely commuting, state-independent baseline."""
    def __init__(self):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(2,2))

    def forward(self, state_logits, edit):
        return state_logits + self.delta[edit]


def labels(rows):
    return torch.tensor([laws(r) for r in rows], dtype=torch.float32)


def source_tensor(values):
    return (values * 15).clamp(-1, 1).detach().clone().float()


def train_decoder(x, y, val_x, val_y, updates, seed):
    torch.manual_seed(seed)
    model=SourceDecoder()
    opt=torch.optim.AdamW(model.parameters(),lr=.014,weight_decay=.001)
    gen=torch.Generator().manual_seed(seed+17)
    best=-1; best_step=0; history=[]; best_weights=None
    for step in range(1,updates+1):
        inds=torch.randint(len(x),(96,),generator=gen)
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(model(x[inds]),y[inds])
        loss.backward();opt.step()
        if step%50==0:
            with torch.no_grad():
                pred=torch.sigmoid(model(val_x))>=.5
                accuracy=(pred==val_y.bool()).all(-1).float().mean().item()
                history.append({"step":step,"train_loss":round(loss.item(),5),
                                "validation_joint_accuracy":accuracy})
                if accuracy>best:
                    best=accuracy;best_step=step
                    best_weights={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    return model,{"best_step":best_step,"validation_joint_accuracy":best,
                  "history":history}


def train_edit(model,updates,seed):
    torch.manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=0)
    gen=torch.Generator().manual_seed(seed+37)
    history=[]
    for step in range(1,updates+1):
        # Atomic supervised laws only, never any two-edit composition.
        bits=torch.randint(0,2,(128,2),generator=gen)
        ops=torch.randint(0,2,(128,),generator=gen)
        desired=torch.tensor(
            [edit_bits(tuple(b),EDITS[int(o)])
             for b,o in zip(bits.tolist(),ops.tolist())],dtype=torch.float32)
        confidence=1+3.5*torch.rand((128,2),generator=gen)
        state=(2*bits.float()-1)*confidence
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(model(state,ops),desired)
        loss.backward();opt.step()
        if step%100==0:
            history.append({"step":step,"atomic_loss":round(loss.item(),5)})
    return model,{"updates":updates,"atomic_loss_curve":history}


def apply_model(model,initial,sequence):
    result=initial
    for edit in sequence:
        indices=torch.full((len(result),),EDITS.index(edit),dtype=torch.long)
        result=model(result,indices)
    return torch.sigmoid(result)


def action_decision(scene,bits):
    # Same host-legal DG004 public action scorer for each counterfactual
    # sequence; terrain and menus never encode private donor truth.
    _,op,_,inv,actions,legal=scene
    h,v=bits
    p=(h*v*gate(op,1,1)+h*(1-v)*gate(op,1,0)
       +(1-h)*v*gate(op,0,1)+(1-h)*(1-v)*gate(op,0,0))
    if inv:p=1-p
    values=[3*(p*a[0]+(1-p)*(1-a[0]))+.38*a[1]+.23*a[2]-.27*a[3]+.1*a[4]
            if legal[j] else -1e8 for j,a in enumerate(actions)]
    return max(range(len(actions)),key=values.__getitem__)


def evaluate(decoder,editor,cases,scores,mode,seed):
    subset=[i for i,r in enumerate(cases) if r["split"]=="sealed"]
    truth0=torch.tensor([laws(cases[i]) for i in subset],dtype=torch.float32)
    source=source_tensor(scores[subset])
    if mode=="zero":
        source=torch.zeros_like(source)
    elif mode=="shuffled":
        rng=random.Random(seed+100)
        permutation=list(range(len(subset)))
        rng.shuffle(permutation)
        source=source[permutation]
    with torch.no_grad():
        initial=(decoder(source) if mode not in ("oracle","symbolic")
                 else (2*truth0-1)*7)
        states={}
        for sequence in SEQUENCES:
            truth=torch.tensor(
                [compose_bits(tuple(map(int,b)),sequence)
                 for b in truth0.tolist()],dtype=torch.float32)
            probs=(truth if mode=="symbolic"
                   else apply_model(editor,initial,sequence))
            states[sequence]=(truth,probs,(probs>=.5).int())
        n=sum(len(x[0]) for x in states.values())
        joint=sum((x[0].int()==x[2]).all(-1).sum().item()
                  for x in states.values())
        bits=sum((x[0].int()==x[2]).sum().item() for x in states.values())
        desired=(states["HV"][0]!=states["VH"][0]).any(-1)
        observed=(states["HV"][2]!=states["VH"][2]).any(-1)
        sceneset=scenes(cases,"sealed",GAME_OPS,4,seed)
        positions={index:pos for pos,index in enumerate(subset)}
        game={"correct":0,"n":0,"legal":0,"expected_order_sensitive":0,
              "observed_order_sensitive":0,"correct_order_flags":0}
        samples=[]
        for scene in sceneset:
            pos=positions[scene[0]]
            choices_gold={};choices_pred={}
            for seq in SEQUENCES:
                true_laws,guess,_=states[seq]
                h,v=true_laws[pos].tolist()
                predicted_laws=guess[pos].tolist()
                gold=target(scene,{"heat":"hardens" if h else "softens",
                                   "vibration":"floats" if v else "sinks"})
                chosen=action_decision(scene,predicted_laws)
                game["correct"]+=gold==chosen
                game["n"]+=1
                game["legal"]+=bool(scene[-1][chosen])
                choices_gold[seq]=gold;choices_pred[seq]=chosen
                if len(samples)<12:
                    samples.append({"case_id":cases[scene[0]]["case_id"],
                                    "sequence":seq,"operator":scene[1],
                                    "turn":scene[2],"true_laws":[h,v],
                                    "predicted_probabilities":predicted_laws,
                                    "correct_action":gold,"selected_action":chosen,
                                    "legal_menu":scene[-1]})
            expected=choices_gold["HV"]!=choices_gold["VH"]
            actual=choices_pred["HV"]!=choices_pred["VH"]
            game["expected_order_sensitive"]+=expected
            game["observed_order_sensitive"]+=actual
            game["correct_order_flags"]+=expected==actual
    game["accuracy"]=game["correct"]/game["n"]
    game["order_flag_accuracy"]=game["correct_order_flags"]/len(sceneset)
    return {"states":n,"joint_state_accuracy":joint/n,
            "bit_accuracy":bits/(2*n),
            "HV_vs_VH":{"paired_cases":len(subset),
                        "truth_change":int(desired.sum()),
                        "predicted_change":int(observed.sum()),
                        "true_positive":int((desired&observed).sum()),
                        "false_positive":int((~desired&observed).sum())},
            "game":game,"sample_receipts":samples}


def selftest():
    assert compose_bits((1,0),"HV")==(1,0)
    assert compose_bits((1,0),"VH")==(0,0)
    states=((0,0),(0,1),(1,0),(1,1))
    assert sum(compose_bits(s,"HV")!=compose_bits(s,"VH")
               for s in states)==3
    baseline=AdditiveEdit()
    state=torch.tensor([[-2.,1.],[.3,-.4]])
    assert torch.allclose(
        baseline(baseline(state,torch.tensor([0,0])),torch.tensor([1,1])),
        baseline(baseline(state,torch.tensor([1,1])),torch.tensor([0,0])))
    samples=torch.tensor(states,dtype=torch.float32)
    candidates=[{"heat":"hardens" if h else "softens",
                 "vibration":"floats" if v else "sinks"}
                for h,v in samples.tolist()]
    assert torch.equal(labels(candidates),samples)
    torch.set_num_threads(2)
    editor,meta=train_edit(ContextEdit(),250,SEED)
    assert meta["updates"]==250
    with torch.no_grad():
        for op in EDITS:
            expected=torch.tensor(
                [edit_bits(tuple(map(int,b)),op)
                 for b in samples.tolist()],dtype=torch.int32)
            actual=(apply_model(editor,(2*samples-1)*3,op)>=.5).int()
            assert torch.equal(expected,actual)
    print("DG005 NONCOMMUTATIVE ATOMIC OPTIMIZER SELFTEST PASSED")


def produce(source,out,seed,updates):
    if not 100<=updates<=2500:
        raise ValueError("bounded optimizer steps: 100..2500")
    torch.set_num_threads(2)
    started=time.monotonic()
    cases,scores,manifest=donor_scores(source)
    partitions={name:[i for i,r in enumerate(cases) if r["split"]==name]
                for name in ("train","validation","sealed")}
    if [len(partitions[k]) for k in partitions] != [80,24,24]:
        raise RuntimeError("DG002 source family split changed")
    x=source_tensor(scores)
    y=labels(cases)
    decoder,dmeta=train_decoder(x[partitions["train"]],y[partitions["train"]],
                                x[partitions["validation"]],y[partitions["validation"]],
                                updates,seed)
    contextual,cmeta=train_edit(ContextEdit(),updates,seed+1)
    additive,ameta=train_edit(AdditiveEdit(),updates,seed+2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoints={}
    for name,model in (("source_decoder",decoder),("context_edit",contextual),
                       ("additive_edit",additive)):
        path=out/(name+".pt")
        torch.save({"state_dict":model.state_dict(),"seed":seed,
                    "source_v_sha256":SOURCE_SHA,"experiment":"DG005"},path)
        checkpoints[name]=digest(path)
    result={name:{mode:evaluate(decoder,model,cases,scores,mode,seed)
                  for mode in ("real","zero","shuffled","oracle","symbolic")}
            for name,model in (("context_edit",contextual),
                               ("additive_edit",additive))}
    report={
        "experiment":"DG005-noncommuting-game-law-alleles-v1",
        "source_run":43736088404,
        "real_qwen_V_sha256":SOURCE_SHA,
        "donor_revision":manifest["donor"]["revision"],
        "source_channel":"supervised two-scalar ridge from real Qwen V; not native KV transfer",
        "source_families_sealed_previously_inspected":True,
        "train_source_families":20,"validation_source_families":6,
        "sealed_source_families":6,
        "trained_edit_examples":"H and V individually, NO composed edit sequences",
        "sealed_compositions":SEQUENCES,
        "public_mechanism":"hand-authored noncommuting truth-table edit laws",
        "environment":"external deterministic CyberSand-like legal actions, NOT native PAINT",
        "checkpoints_sha256":checkpoints,
        "decoder_training":dmeta,"context_training":cmeta,
        "additive_training":ameta,"results":result,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "limitations":["two supervised source scalars","atomic edit supervision",
                       "no native heterogeneous KV transfer",
                       "no fresh independent source family corpus",
                       "scripted action projection, not trained action head",
                       "no CyberSand code/content modification"]}
    path=out/"report.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"experiment":report["experiment"],
                      "elapsed_seconds":report["elapsed_seconds"],
                      "state_joint":{n:{m:round(z["joint_state_accuracy"],3)
                                         for m,z in v.items()}
                                     for n,v in result.items()},
                      "game_accuracy":{n:{m:round(z["game"]["accuracy"],3)
                                           for m,z in v.items()}
                                       for n,v in result.items()}},indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--source")
    p.add_argument("--out")
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--updates",type=int,default=600)
    args=p.parse_args()
    if args.selftest:
        selftest()
    else:
        if not args.source or not args.out:
            p.error("--source and --out required")
        produce(args.source,args.out,args.seed,args.updates)


if __name__=="__main__":
    main()
