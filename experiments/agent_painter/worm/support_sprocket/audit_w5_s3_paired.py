#!/usr/bin/env python3
"""Source-fenced independent W5/S3 vs W4/S2 model-score and W0 pixel audit.

The original model outputs and Python-W0 PNGs are checked, never regenerated
from teacher targets. No optimizer, model download, Godot, or FINAL access.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import zipfile
import zlib

ACTIONS = ("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP")
FAMILIES = ("greebles","pipework","architecture")
SEEDS = (211,223)
CONTROLS = ("random","wall","fidget")
SOURCE = {
    "s2": ("014818c9cb44826aafb6bb0124228addcc0a602f3ff379b14b284ae17ddfb811",
           "d0d34ab031461aa38e15b24a64bc6c1fb64cb882"),
    "s3": ("a994920e4dc21ec36e47c63a553b0ac6729947399be7cb2301c09b98d6e7c6aa",
           "39568337f2b2e278581151e5694e77d1a7942cc8"),
}
W0_SHA = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
W0_COMMIT = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_zip(path, label):
    expected, source = SOURCE[label]
    assert sha(path.read_bytes()) == expected, (label,"ZIP hash")
    z = zipfile.ZipFile(path)
    assert z.testzip() is None
    entries = set(n for n in z.namelist() if not n.endswith("/"))
    inventory = {}
    for line in z.read("SHA256SUMS.txt").decode().splitlines():
        value, name = line.split("  ",1)
        name = name.removeprefix("./")
        assert name not in inventory and len(value)==64 and name in entries
        inventory[name] = value
    assert set(inventory)==entries-{"SHA256SUMS.txt"}
    for name, value in inventory.items():
        assert sha(z.read(name)) == value, (label,name)
    assert z.read("source-sha.txt").decode().strip()==source
    return z, len(entries), len(inventory)


def pin_w0(path):
    assert sha(path.read_bytes())==W0_SHA
    spec=importlib.util.spec_from_file_location("original_worm_support_s3", path)
    w0=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w0)
    assert tuple(w0.ACTIONS)==ACTIONS and tuple(w0.FAMILIES)==FAMILIES
    return w0


def uniform(seed,tick):
    message=f"612:sprocket:score-crn-s2:{seed}:{tick}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8],"big")/(1<<64)


def sample(logits,legal,temperature,u):
    assert set(logits)==set(ACTIONS) and legal and len(legal)==len(set(legal))
    assert set(legal)<=set(ACTIONS) and temperature in (1.,3.) and 0<=u<1
    candidates=[a for a in ACTIONS if a in legal]
    scaled=[float(logits[a])/temperature for a in candidates]
    assert all(math.isfinite(x) for x in scaled)
    weights=[math.exp(v-max(scaled)) for v in scaled]
    total=sum(weights)
    probs={a:0.0 for a in ACTIONS}
    for a,v in zip(candidates,weights):
        probs[a]=v/total
    cumulative=0.0
    choice=candidates[-1]
    for a in candidates:
        cumulative+=probs[a]
        if u<cumulative:
            choice=a
            break
    flat=candidates[min(int(u*len(candidates)),len(candidates)-1)]
    entropy=-sum(v*math.log(v) for v in probs.values() if v>0)
    effective=1/sum(v*v for v in probs.values() if v>0)
    return choice,flat,probs,entropy,effective


def decode_w0_png(blob):
    """Only the immutable W0 PNG shape is accepted (RGB8, filter 0)."""
    assert blob[:8]==b"\x89PNG\r\n\x1a\n"
    pos=8
    compressed=[]
    while pos<len(blob):
        length=struct.unpack(">I",blob[pos:pos+4])[0]
        name=blob[pos+4:pos+8]
        payload=blob[pos+8:pos+8+length]
        crc=struct.unpack(">I",blob[pos+8+length:pos+12+length])[0]
        assert zlib.crc32(name+payload)&0xffffffff==crc
        pos+=length+12
        if name==b"IHDR":
            assert struct.unpack(">IIBBBBB",payload)==(384,384,8,2,0,0,0)
        if name==b"IDAT":
            compressed.append(payload)
        if name==b"IEND":
            break
    raw=zlib.decompress(b"".join(compressed))
    stride=384*3+1
    assert len(raw)==stride*384
    assert all(raw[y*stride]==0 for y in range(384))
    return [raw[y*stride+1:(y+1)*stride] for y in range(384)]


def png_from_rows(rows):
    width=len(rows[0])//3
    assert width and all(len(r)==width*3 for r in rows)
    def chunk(tag,data):
        return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data)&0xffffffff)
    return (b"\x89PNG\r\n\x1a\n"
            +chunk(b"IHDR",struct.pack(">IIBBBBB",width,len(rows),8,2,0,0,0))
            +chunk(b"IDAT",zlib.compress(b"".join(b"\x00"+r for r in rows),9))
            +chunk(b"IEND",b""))


def comparison_gallery(originals,path):
    # Exact unresampled original W0 pixels, cropped around x=7,y=48.
    panels=[decode_w0_png(b) for b in originals]
    gutter=bytes((39,43,53))*8
    rows=[gutter.join(p[y][0:256*3] for p in panels) for y in range(80,304)]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(png_from_rows(rows))


def run(s2,s3,w0,out):
    archives={"s2":s2,"s3":s3}
    meta={key:json.loads(z.read("results/comparison.json")) for key,z in archives.items()}
    assert meta["s2"]["schema"]=="cybersand.worm.sprocket-live-crn-ablation-s2"
    assert meta["s3"]["schema"]=="cybersand.worm.sprocket-discriminative-live-eval-s3"
    assert meta["s3"]["training_source"]=="984ef5fd172c2db50ca1f3b69678091834fb75d3"
    assert meta["s3"]["training_run"]==43732897139
    for label,m in meta.items():
        assert m["w0_commit"]==W0_COMMIT
        assert m["model"]=="kandivault/sprocket-500m"
        assert m["revision"]=="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
        assert m["actual_optimizer_steps_this_job"]==0
        assert m["native"] is False and m["sealed_final_opened"] is False
        assert m["completed"] is True and m["seed_groups"]==list(SEEDS)
        assert m["families"]==list(FAMILIES) and m["steps_per_trajectory"]==24
        assert m["model_rollouts"]==12 and m["controls"]==18 and len(m["rows"])==30
    assert meta["s3"]["no_off_policy_score_tape"] and meta["s3"]["same_observation_uniform_ablation"]
    maps={v:{(r["family"],r["seed"],r["mode"]):r for r in m["rows"]}
          for v,m in meta.items()}
    assert all(len(m)==30 for m in maps.values())
    results={v:Counter() for v in archives}
    images={}
    columns=["random","wall","fidget","W4-S2-T1","W5-S3-T1","W4-S2-T3","W5-S3-T3"]
    pairs=[]
    controls_verified=0
    replayed=0
    for family in FAMILIES:
        for seed in SEEDS:
            for version,z in archives.items():
                modes=list(CONTROLS)+(
                    ["sprocket-scored-t1","sprocket-scored-t3"] if version=="s2"
                    else ["sprocket-v5-scored-t1","sprocket-v5-scored-t3"])
                for mode in modes:
                    key=(family,seed,mode)
                    name=f"results/png-and-json/{family}-{mode}-{seed}"
                    rec=json.loads(z.read(name+".json"))
                    png=z.read(name+".png")
                    images[(version,*key)]=png
                    assert (rec["family"],rec["seed"],rec["mode"])==key
                    row=maps[version][key]
                    assert rec["metrics"]==row["metrics"]
                    assert len(rec["events"])==rec["steps_executed"]<=24
                    actions=[e["raw_choice"] for e in rec["events"]]
                    replay=w0.run(seed,"replay",family,steps=24,raw_actions=actions)
                    for namefield in ("events","trajectory","obstacles","branch_points","metrics"):
                        assert replay[namefield]==rec[namefield],(version,key,namefield)
                    assert w0.png_bytes(replay)==png,(version,key,"PNG")
                    replayed+=1
                    pts=[tuple(p) for p in rec["trajectory"]]
                    xx,yy=zip(*pts)
                    x0,y0=pts[0]
                    spatial={"unique_cells":len(set(pts)),
                        "bbox_width":max(xx)-min(xx)+1,
                        "bbox_height":max(yy)-min(yy)+1,
                        "max_manhattan_from_start":max(abs(x-x0)+abs(y-y0) for x,y in pts)}
                    assert row["spatial"]==spatial
                    if mode in CONTROLS:
                        other=archives["s3" if version=="s2" else "s2"]
                        assert png==other.read(f"results/png-and-json/{family}-{mode}-{seed}.png")
                        controls_verified+=1
                        continue
                    t=1. if mode.endswith("t1") else 3.
                    receipts=rec["score_receipts"]
                    assert len(receipts)==len(actions)==row["sampled_decisions"]
                    result=results[version]
                    result["rollouts"]+=1
                    result["unique_sum"]+=spatial["unique_cells"]
                    result["reach_sum"]+=spatial["max_manhattan_from_start"]
                    result["bbox_area_sum"]+=spatial["bbox_width"]*spatial["bbox_height"]
                    result["early_stops"]+=int(len(receipts)<24)
                    result[f"T{int(t)}-unique_sum"]+=spatial["unique_cells"]
                    result[f"T{int(t)}-early_stops"]+=int(len(receipts)<24)
                    result[f"T{int(t)}-rollouts"]+=1
                    for tick,r in enumerate(receipts):
                        observation=rec["events"][tick]["observation"]
                        assert r["tick"]==tick and r["host_legal"]==observation["legal"]
                        assert r["selected_action"]==actions[tick]==r["applied_by_host"]
                        assert rec["events"][tick]["applied"]==actions[tick]
                        assert rec["events"][tick]["raw_legal"] is True
                        assert r["temperature"]==t and r["is_model_logit_conditioned"]
                        assert r["is_eos5_completion"] is False and r["is_greedy_decoding"] is False
                        u=uniform(seed,tick)
                        assert abs(u-r["host_uniform_draw"])<1e-15
                        selected,flat,probs,entropy,effective=sample(
                            r["full_six_token_logits"],observation["legal"],t,u)
                        assert selected==actions[tick]
                        assert flat==r["paired_uniform_same_observation"]
                        assert (selected!=flat)==r["paired_differs_from_uniform"]
                        assert all(abs(probs[a]-r["masked_probs"][a])<1e-9 for a in ACTIONS)
                        top=max(ACTIONS,key=lambda a:r["full_six_token_logits"][a])
                        assert r["raw_unmasked_top_action"]==top
                        if version=="s3":
                            assert len(r["action_token_ids"])==6
                            expected_obs=sha(json.dumps(observation,sort_keys=True,separators=(",",":")).encode())
                            assert r["observation_sha256"]==expected_obs
                            assert abs(entropy-r["entropy_nats"])<1e-9
                            assert abs(effective-r["effective_actions"])<1e-9
                        result["decisions"]+=1
                        result["top_"+top]+=1
                        result["selected_"+selected]+=1
                        result["paired_differences"]+=int(selected!=flat)
                    assert row["paired_diff_from_uniform"]==sum(r["paired_differs_from_uniform"] for r in receipts)
            order=[images[("s3",family,seed,m)] for m in CONTROLS]
            order += [images[("s2",family,seed,"sprocket-scored-t1")],
                      images[("s3",family,seed,"sprocket-v5-scored-t1")],
                      images[("s2",family,seed,"sprocket-scored-t3")],
                      images[("s3",family,seed,"sprocket-v5-scored-t3")]]
            comparison_gallery(order,out/"gallery"/f"{family}-{seed}-S2-S3-original-crop.png")
            pairs.append({"family":family,"seed":seed,"columns":columns})
    assert replayed==60 and controls_verified==36
    for v in results:
        assert results[v]["rollouts"]==12
        assert results[v]["decisions"]==meta[v]["paired_uniform_action_count"]
    diversity={}
    for version in archives:
        prefix="sprocket-scored-t" if version=="s2" else "sprocket-v5-scored-t"
        for t in (1,3):
            digests={sha(images[(version,f,s,prefix+str(t))])
                     for f in FAMILIES for s in SEEDS}
            diversity[f"{version}_T{t}_unique_PNGs_among_six_labelled_views"]=len(digests)
    return {
       "schema":"cybersand.worm.sprocket-support-S3-paired-original-v1",
       "inputs":{v:{"sha256":SOURCE[v][0],"producer_commit":SOURCE[v][1]} for v in SOURCE},
       "W0_sha256":W0_SHA,"original_full_W0_PNG_and_JSON_replays":replayed,
       "identical_original_control_PNG_checks":controls_verified,
       "score_statistics":{v:dict(results[v]) for v in results},
       "source_PNG_diversity":diversity,"gallery_manifest":pairs,
       "optimizer_updates":0,"Godot_native_paintings":0,
       "limitations":["two independent obstacle worlds, three correlated family-labelled variants",
                      "sampled live logits, not greedy decoded EOS completions",
                      "uniform paired action is a same-observation ablation, not a closed-loop uniform rollout",
                      "gallery crops are unresampled original W0 pixels"]
    }


def self_test():
    flat={a:0.0 for a in ACTIONS}
    for u in (0.0,.001,.499,.9,.999):
        action,control,probs,entropy,effective=sample(flat,["LEFT","STOP"],1.,u)
        assert action==control and abs(sum(probs.values())-1)<1e-12
        assert 1<=effective<=2 and entropy>=0
    assert uniform(211,0)!=uniform(211,1)
    assert uniform(211,0)!=uniform(223,0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s2",type=Path)
    p.add_argument("--s3",type=Path)
    p.add_argument("--trusted-w0",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test()
    if args.self_test:
        print("S3 finite legal-action probability and RNG fixtures PASS")
        return
    assert args.s2 and args.s3 and args.trusted_w0 and args.output
    z2,n2,h2=read_zip(args.s2,"s2")
    z3,n3,h3=read_zip(args.s3,"s3")
    report=run(z2,z3,pin_w0(args.trusted_w0),args.output.parent)
    report["original_ZIP_CRC_counts"]={"s2":n2,"s3":n3}
    report["original_member_SHA256_counts"]={"s2":h2,"s3":h3}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("gallery_manifest","limitations")},sort_keys=True))

if __name__=="__main__":
    main()
