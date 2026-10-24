#!/usr/bin/env python3
import json,sys
from collections import Counter,defaultdict
from pathlib import Path

out=Path(sys.argv[1]); files=[Path(x) for x in sys.argv[2:]]
records=[json.loads(p.read_text()) for p in files if p.exists()]
summary={"experiment":"panel-001","models":[],"finalist_votes":{},"verdicts":{},"score_means":{},"recombinations":[]}
votes=Counter(); verdicts=defaultdict(Counter); scores=defaultdict(lambda:defaultdict(list))
for r in records:
    p=r.get("parsed")
    summary["models"].append({"model_id":r["model_id"],"parsed":p is not None,"tokens_per_second":r.get("tokens_per_second"),"rss_kib":r.get("server_rss_kib")})
    if not p: continue
    for cid in p.get("finalists",[]): votes[cid]+=1
    for cid,review in p.get("reviews",{}).items():
        verdicts[cid][review.get("verdict","?")]+=1
        for k,v in (review.get("scores") or {}).items():
            if isinstance(v,(int,float)): scores[cid][k].append(v)
    summary["recombinations"].append({"model_id":r["model_id"],**(p.get("recombination") or {})})
summary["finalist_votes"]=dict(votes.most_common())
summary["verdicts"]={k:dict(v) for k,v in verdicts.items()}
summary["score_means"]={cid:{k:sum(v)/len(v) for k,v in dims.items() if v} for cid,dims in scores.items()}
out.mkdir(parents=True,exist_ok=True)
(out/"panel-001-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# Panel 001 aggregate","","## Finalist votes",""]
for cid,n in votes.most_common(): md.append(f"- {cid}: {n}")
md += ["","## Per-model",""]
for m in summary["models"]: md.append(f"- {m['model_id']}: parsed={m['parsed']}, {m['tokens_per_second']:.2f} tok/s" if m["tokens_per_second"] else f"- {m['model_id']}: parsed={m['parsed']}")
md += ["","## Verdict distributions",""]
for cid in sorted(verdicts): md.append(f"- {cid}: {dict(verdicts[cid])}")
md += ["","## Recombinations",""]
for x in summary["recombinations"]: md += [f"### {x.get('model_id')}","",f"Parents: {x.get('parents')}","",x.get("mechanism",""),"",f"Failure: {x.get('failure','')}",""]
(out/"panel-001-summary.md").write_text("\n".join(md)+"\n")
print("\n".join(md))
