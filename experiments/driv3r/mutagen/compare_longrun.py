#!/usr/bin/env python3
"""Read-only post-run comparison of actual per-arm d1 M1 artifacts.

Checks that W0 initial world/seed/family observations match. Builds a contact
sheet ONLY from original model-in-loop PNGs; does not rerun inference, repair
choices, synthesize missing models or claim native Godot material readback.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ARMS=("stock","mlp_scoop","attn_gouge","conv_scrape_10","conv_scrape_25")
SEEDS=(570,2111)
FAMILIES=("greebles","pipework","architecture")


def load(root,sha):
    results={}
    for arm in ARMS:
        folder=root / f"driv3r-d1-m1-{arm}-{sha}"
        entry={"arm":arm,"artifact_exists":folder.is_dir(),"folder":str(folder),
               "has_summary":False,"status":"MISSING"}
        file=folder/"run"/"summary.json"
        if file.is_file():
            data=json.loads(file.read_text())
            if data.get("arm")!=arm or data.get("schema")!="driv3r-d1-mutagen-longrun/1":
                raise ValueError("Wrong original M1 artifact arm/schema")
            entry.update({"has_summary":True,"status":data["status"],"data":data})
        results[arm]=entry
    return results


def compare(root,sha,out):
    out.mkdir(parents=True,exist_ok=True)
    results=load(root,sha)
    summaries={}
    matched=[]
    for seed in SEEDS:
        for family in FAMILIES:
            row={"seed":seed,"family":family,"arms":{}}
            reference=None
            reference_actions=[]
            for arm in ARMS:
                data=results[arm].get("data")
                episodes=([e for e in data["episodes"]
                           if e["seed"]==seed and e["family"]==family]
                          if data else [])
                if len(episodes)>1:
                    raise ValueError("Duplicate same-arm same-world evidence")
                item=episodes[0] if episodes else None
                if item:
                    initial=item["initial_observation_sha256"]
                    if reference is None:
                        reference=initial
                    elif initial!=reference:
                        raise ValueError("Mismatched starting W0 observations across arms")
                    dec=item["decisions"]
                    row["arms"][arm]={
                      "completed":item["done"],
                      "actual_decisions":dec["valid_typed_decisions"],
                      "unique_cells":item.get("trajectory_metrics",{}).get("unique_trail_cells"),
                      "branches":item.get("trajectory_metrics",{}).get("branches"),
                      "revisits":item.get("trajectory_metrics",{}).get("total_revisit_contacts"),
                      "stop":dec["stop_selected"],
                      "action_histogram":dec["action_histogram"],
                      "mean_model_seconds":dec["mean_model_latency_seconds"],
                      "first_16_raw_actions":dec["raw_choices"][:16],
                    }
                    if arm=="stock":
                        reference_actions=dec["raw_choices"]
                else:
                    row["arms"][arm]={"completed":False,"reason":"missing original summary/episode"}
            if reference_actions:
                for arm in ARMS[1:]:
                    value=row["arms"][arm]
                    x=results[arm].get("data")
                    candidate=next((e["decisions"]["raw_choices"] for e in x["episodes"]
                                    if e["seed"]==seed and e["family"]==family),[]) if x else []
                    upto=min(len(reference_actions),len(candidate))
                    first=next((i for i in range(upto)
                                if reference_actions[i]!=candidate[i]),None)
                    value["first_raw_action_divergence_at_tick"] = first
                    value["prefix_equal_length"] = first if first is not None else upto
                    value["compared_common_decisions"] = upto
            row["initial_observation_sha256"]=reference
            matched.append(row)
    for arm,source in results.items():
        data=source.get("data")
        episodes=[e for e in data["episodes"]] if data else []
        done=[e for e in episodes if e["done"]]
        summaries[arm]={
          "status":source["status"],"summary_available":source["has_summary"],
          "completed_episodes":len(done),"expected_episodes":6,
          "total_valid_real_model_choices":sum(
              e["decisions"]["valid_typed_decisions"] for e in episodes),
          "unique_cell_totals":sum(
              e["trajectory_metrics"]["unique_trail_cells"] for e in done),
          "stop_count":sum(e["decisions"]["stop_selected"] for e in episodes),
          "action_histogram":data["overall_action_histogram"] if data else None,
          "p50_model_seconds":data["p50_model_seconds"] if data else None,
          "p95_model_seconds":data["p95_model_seconds"] if data else None,
        }
    comparison={"schema":"driv3r-d1-mutagen-longrun-comparison/1",
       "exact_request_sha":sha,
       "expected_arms":list(ARMS),"seeds":list(SEEDS),"families":list(FAMILIES),
       "all_arms_complete":all(source["status"]=="COMPLETE" for source in results.values()),
       "matched_arms":summaries,
       "identical_initial_W0_observations_confirmed":all(
           all(a["completed"] for a in row["arms"].values())
           and row["initial_observation_sha256"] is not None
           for row in matched),
       "episode_comparisons":matched,
       "note":"Only identical *starting* worlds and source W0 semantics are matched. Later observations differ after model choices.",
       "not_native":True,"model_training":False,
       "missing_or_partial_included":True}
    (out/"comparison.json").write_text(json.dumps(comparison,indent=2,sort_keys=True)+"\n")
    return comparison


def gallery(root,sha,out,comparison):
    from PIL import Image,ImageDraw,ImageFont
    cell=220
    left=144
    top=94
    rowh=276
    colw=252
    width=left+len(ARMS)*colw+20
    height=top+len(SEEDS)*len(FAMILIES)*rowh+22
    image=Image.new("RGB",(width,height),(21,25,32))
    draw=ImageDraw.Draw(image)
    try:
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
        small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",12)
    except OSError:
        font=ImageFont.load_default()
        small=font
    draw.text((14,10),"d1-2B M1: SOURCE-FAITHFUL 128/64-decision real model worms",
              fill=(242,245,255),font=font)
    draw.text((14,38),"Same W0 seed and theme across arms | original model outputs, NOT native Godot material",
              fill=(170,186,207),font=small)
    for col,arm in enumerate(ARMS):
        draw.text((left+col*colw+4,68),arm,fill=(226,224,201),font=small)
    for index,row in enumerate(comparison["episode_comparisons"]):
        y=top+index*rowh
        draw.text((9,y+3),f'{row["family"]}',fill=(239,239,248),font=small)
        draw.text((9,y+22),f'seed {row["seed"]}',fill=(164,176,193),font=small)
        for col,arm in enumerate(ARMS):
            x=left+col*colw
            entry=row["arms"][arm]
            original=root/f"driv3r-d1-m1-{arm}-{sha}"/"run"/"episodes"/f"{row['family']}-d1-m1-{arm}-{row['seed']}.png"
            if original.is_file() and entry.get("completed"):
                with Image.open(original) as png:
                    raster=png.convert("RGB").resize((cell,cell),Image.Resampling.NEAREST)
                image.paste(raster,(x,y))
                draw.text((x,y+222),f'{entry["actual_decisions"]} decisions | {entry["unique_cells"]} cells',
                          fill=(223,230,242),font=small)
                draw.text((x,y+239),
                          f'B{entry["branches"]} R{entry["revisits"]} STOP={int(entry["stop"])}',
                          fill=(169,191,199),font=small)
            else:
                draw.rectangle((x,y,x+cell,y+cell),fill=(48,39,43))
                draw.text((x+10,y+20),"INCOMPLETE / FAILED",fill=(255,176,159),font=small)
                draw.text((x,y+225),str(entry.get("reason","no complete PNG"))[:27],
                          fill=(215,170,165),font=small)
    path=out/"real-model-contact-sheet.png"
    image.save(path)
    comparison["gallery_png_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"comparison.json").write_text(json.dumps(comparison,indent=2,sort_keys=True)+"\n")
    return path


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--sha",required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=compare(args.root,args.sha,args.out)
    path=gallery(args.root,args.sha,args.out,result)
    print(json.dumps({"complete":result["all_arms_complete"],
                      "arms":list(result["matched_arms"]),
                      "gallery":str(path)}),flush=True)
    if not result["all_arms_complete"]:
        raise SystemExit("One or more long-run arms incomplete; originals preserved")
