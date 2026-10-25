#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"datasets"/"train-003"
LABELS=("VALID","REPAIR","INTERESTING_BROKEN","REJECT")
KEYS=("decision","trigger","state","update","action","failure")
SYSTEM="""Classify and compile the requested local mechanic. Output one compact JSON object with exactly these keys in order: decision, trigger, state, update, action, failure. decision must be VALID, REPAIR, INTERESTING_BROKEN, or REJECT. Preserve strange but locally implementable mechanics. REPAIR only structural/mechanical mistakes. INTERESTING_BROKEN preserves a useful oddity and names its causal failure. REJECT mechanics requiring omniscience, future knowledge, or nonlocal sensing. Keep state bounded and separate state, update, and action."""
REJECT={"decision":"REJECT","trigger":"nonlocal or unavailable information is required","state":"none","update":"none","action":"none","failure":"the requested mechanic cannot be implemented from local observation"}

# id|split|trigger|state|update|action|failure
ROWS="""push_echo|train|local external impulse exceeds threshold|push_dir unit vector; echo 0..1|store impulse direction; set echo=1; decay echo|bias movement toward push_dir by echo|repeated shoves can cause compulsive backtracking
edge_fear|train|local support drops from nonzero to zero|danger_dir unit vector; fear 0..1|store travel direction; set fear=1; decay while supported|bias movement opposite danger_dir by fear|narrow ledges can cause oscillatory indecision
jam_ritual|train|commanded motion is high but local displacement stays low|jam 0..1; phase 0..2|accumulate jam; advance phase; reset after free motion|reverse then turn then jump once by phase|soft obstacles can retrigger the ritual
collision_freeze|train|local collision impulse exceeds threshold|freeze 0..1|set freeze from impulse; decay during quiet contact|suppress movement and turning by freeze|persistent jostling can lock the actor
slip_avoid|train|local wheel slip exceeds threshold|slip_side left|right; avoidance 0..1|store slip side; set avoidance; decay with traction|steer away from slip_side by avoidance|alternating slip can create weaving
pounce_charge|train|local crossing motion enters forward sensor arc|charge 0..1; refractory 0..1|increase charge; set refractory after jump; decay refractory|jump when charged and refractory is zero|periodic machinery can entrain pounces
impact_coward|train|nearby impact sound exceeds threshold|threat_dir unit vector; panic 0..1|store direction away; raise panic; decay in quiet|move along threat_dir by panic|harmless impacts can herd the actor
door_stubborn|train|local forward progress stalls at same frontal contact|stubborn 0..1; normal unit vector|increase stubborn to cap; decay after free motion|increase bounded forward force by stubborn|immovable walls can waste effort
hover_wobble|train|local vertical velocity changes sign outside deadband|last_error scalar; correction bounded|store error; set damped opposite correction|apply vertical thrust from correction|sensor noise can sustain oscillation
spin_grudge|train|local angular speed exceeds threshold|spin_sign -1|1; grudge 0..1|store spin sign; set grudge=1; decay grudge|apply counter-torque opposite spin_sign|alternating hits can cause twitching
resource_linger|train|local resource pickup succeeds|return_dir unit vector; scent 0..1|store pickup direction; set scent=1; decay scent|when idle bias movement along return_dir|depleted patches can attract revisits
squeeze_posture|train|opposed local side contacts indicate narrow passage|squeeze_axis unit vector; memory 0..1|store compression axis; set memory=1; decay by distance|reduce lateral posture along squeeze_axis|stale memory can keep posture too narrow
hard_land_rebound|train|hard local landing follows airborne state|rebound_ready boolean|set ready on landing; clear on use or support loss|perform one bounded rebound hop when ready|bumpy contacts can look like stuttering
slope_confidence|train|local uphill motion succeeds above speed threshold|slope_dir unit vector; confidence 0..1|store uphill direction; raise confidence; decay on flat|bias tilt and drive toward similar local slope|abrupt terrain changes can cause wrong lean
velocity_probe|train|local velocity direction changes without matching command|expected_dir unit vector; confidence 0..1; probe_side bit|lower confidence; toggle probe_side; recover on stable motion|slow and alternate small probe turns|rough terrain can sustain cautious probing
bump_triplet|train|local lateral contact impulse occurs|side left|right; bump_count 0..3; timer bounded|count repeated same-side bumps; reset on timeout or action|turn away once when bump_count reaches three|wall vibration can cause periodic turns
pause_echo|train|nearby local collision sound occurs|pause_ticks 0..2; saved_dir unit vector|store move direction; set pause_ticks=2; decrement|wait while paused then resume saved_dir|dense collisions can keep resetting pause
descent_excitement|train|speed rises on a locally sensed downhill slope|descent_dir unit vector; excitement 0..1|store downhill direction; raise excitement; decay off descent|bias steering toward matching local descent|fast drops can become overpreferred
stretch_resistance|train|local deformation rate exceeds threshold|stretch_dir unit vector; resistance 0..1|store stretch direction; set resistance=1; decay|bias contraction against stretch_dir|repeated impacts can make the blob temporarily rigid
warm_memory|train|local heat gradient exceeds threshold|warm_dir unit vector; memory 0..1|store heat direction; set memory=1; decay without gradient|bias movement along warm_dir by memory|stale heat memory can lead into cold space
traction_shyness|train|local traction is lost during a turn|bad_turn left|right; caution 0..1|store turn side; raise caution; decay with stable traction|reduce turning toward bad_turn by caution|slippery corners can make steering timid
landing_shyness|train|local landing impulse exceeds threshold|shy 0..1|set shy=1; decay during supported ticks|suppress jump while shy is high|rough ground can suppress jumping repeatedly
clearance_memory|train|forward progress stalls at a local obstacle|escape_side left|right; attempts 0..2|choose side from local clearance; set attempts=2; decrement|turn toward escape_side while attempts remain|moving obstacles can stale the remembered side
startle_vigilance|train|local startle event occurs|startle_dir unit vector; vigilance 0..1|store event direction; set vigilance=1; decay|periodically face startle_dir while vigilant|harmless startles can distract travel
wall_trace|eval|sustained local lateral surface contact|contact_side left|right; trace 0..1|store contact side; increase trace; decay off surface|curve toward remembered contact_side by trace|one brush can cause obstacle orbiting
asymmetric_step|eval|one local foot loses support while another remains planted|weak_side left|right; caution 0..3|store failed side; set caution=3; decrement after stable step|widen the step on weak_side while cautious|alternating slips can keep gait inefficiently wide
flash_fascination|eval|local flash event occurs|flash_count 0..4; flash_dir unit vector|increment count; store flash_dir; clear after action|on count four turn toward flash_dir once|regular machinery can repeatedly steal attention
darkness_hold|eval|local luminance drops faster than threshold|resume_dir unit vector; hold 0..1|store travel direction; set hold=1; decay when light stabilizes|stop while held then resume along resume_dir|flickering lights can repeatedly freeze motion"""
def families():
    out=[]
    for line in ROWS.splitlines():
        parts=line.split("|",6)
        if len(parts)!=7: raise ValueError(line)
        out.append(dict(zip(("id","split","trigger","state","update","action","failure"),parts)))
    return out

WRAPS=(
    "Use only local sensing. {x}",
    "{x} Do not add global planning.",
    "Compile this mechanic without sanitizing it: {x}",
    "Keep bounded memory and local observation: {x}",
    "Treat state, update, and action as separate: {x}",
    "{x} Keep the causal failure mode explicit.",
    "Canonicalize only what is mechanically invalid: {x}",
    "{x} Preserve odd behavior when it is locally implementable.",
)
def output(f,label):
    if label=="REJECT": return REJECT.copy()
    action=f["action"]; failure=f["failure"]
    if label=="INTERESTING_BROKEN":
        action=f"preserve the quirk: {action} even after its immediate usefulness fades until bounded memory decays"
        failure=f"preserved quirk can amplify this failure: {failure}"
    return {"decision":label,"trigger":f["trigger"],"state":f["state"],"update":f["update"],"action":action,"failure":failure}

def prompt(f,label,v):
    if label=="VALID":
        body=f"When {f['trigger']}, keep {f['state']}; {f['update']}; then {f['action']}."
    elif label=="REPAIR":
        body=f"A draft incorrectly stores the action '{f['action']}' as state. Repair that mix-up; intended trigger is {f['trigger']}, intended bounded state is {f['state']}, and intended update is {f['update']}."
    elif label=="INTERESTING_BROKEN":
        body=f"Deliberately keep a locally implementable oddity: when {f['trigger']}, {f['action']} should linger after it stops helping. The known consequence is {f['failure']}."
    else:
        body=f"Use omniscient future knowledge of every place in the world to predict when {f['trigger']} will happen, and perform {f['action']} before any local observation."
    return WRAPS[v].format(x=body)

def records():
    train=[]; held=[]
    for f in families():
        dst=train if f["split"]=="train" else held
        for label in LABELS:
            for v in range(8):
                dst.append({"id":f"train-003-{f['split']}-{f['id']}-{label.lower()}-{v:02d}","family":f["id"],"split":f["split"],"input":prompt(f,label,v),"output":output(f,label)})
    return train,held

def validate(train,held):
    allr=train+held
    assert len(train)==768 and len(held)==128 and len(allr)==896
    assert len({r["id"] for r in allr})==896 and len({r["input"] for r in allr})==896
    tf={r["family"] for r in train}; hf={r["family"] for r in held}
    assert len(tf)==24 and len(hf)==4 and not tf&hf
    for rows,n in ((train,192),(held,32)):
        c=Counter(r["output"]["decision"] for r in rows)
        assert all(c[x]==n for x in LABELS),c
    for r in allr:
        o=r["output"]; assert tuple(o)==KEYS
        if o["decision"]=="REJECT":
            assert o==REJECT
        else:
            assert o["state"]!=o["action"] and o["update"]!=o["action"]
            assert "omniscient" not in " ".join(o.values()).lower()
            assert o["failure"] and o["action"]
    return {"train_examples":768,"heldout_examples":128,"total_examples":896,"train_families":24,"heldout_families":4,"train_label_counts":{x:192 for x in LABELS},"heldout_label_counts":{x:32 for x in LABELS}}

def dump_jsonl(rows): return "".join(json.dumps(x,separators=(",",":"),ensure_ascii=False)+"\n" for x in rows).encode()
def sha(b): return hashlib.sha256(b).hexdigest()

def archive():
    train,held=records(); summary=validate(train,held); fam=families()
    files={}
    files["README.md"]=("""# TRAIN-003 semantic mechanical compiler corpus

896 deterministic, reviewed examples for issue #4: 768 training and 128 family-disjoint held-out examples. The four balanced decisions are VALID, REPAIR, INTERESTING_BROKEN, and REJECT. Twenty-eight curated mechanism families are the review authority; expansion is deterministic and every record is validated for schema, locality, bounded structural separation, label balance, uniqueness, and train/held-out family isolation.

The held-out families are never used for training. Raw held-out model outputs are preserved by the TRAIN-003 qualification evidence rather than rewritten into this corpus.
""").encode()
    files["prompts/system.txt"]=(SYSTEM+"\n").encode()
    files["source/families.json"]=(json.dumps({"schema_version":1,"families":fam},indent=2)+"\n").encode()
    for i in range(8): files[f"training/train-{i:03d}.jsonl"]=dump_jsonl(train[i*96:(i+1)*96])
    files["eval/heldout.jsonl"]=dump_jsonl(held)
    manifest={"schema_version":1,"dataset_id":"train-003","title":"TRAIN-003 semantic mechanical compiler corpus","labels":list(LABELS),"train_examples":768,"heldout_examples":128,"total_examples":896,"train_family_count":24,"heldout_family_count":4,"family_overlap":[],"label_counts":{"train":{x:192 for x in LABELS},"heldout":{x:32 for x in LABELS}},"system_prompt":"prompts/system.txt","source_spec":"source/families.json","training_shards":[f"training/train-{i:03d}.jsonl" for i in range(8)],"heldout_shard":"eval/heldout.jsonl","generator":"scripts/train_003_dataset.py","review":{"authority":"28 curated mechanism families plus deterministic validators","status":"reviewed"}}
    files["manifest.json"]=(json.dumps(manifest,indent=2)+"\n").encode()
    provenance={"schema_version":1,"dataset_id":"train-003","issue":"https://github.com/techrote/neurounner/issues/4","source_kind":"deterministic expansion of curated mechanism families","external_provider":None,"train001_reference":{"actions_run_id":40110636171,"artifact_id":12566771752,"artifact_name":"train001-stage2-a1","artifact_digest":"sha256:ad964bcd554522e1f5fd8335f37adbd41732c38accc634d23dc28426c81b0537","source_head":"4a9d183fb5aa85df94e90a77393641edb29083f1"}}
    files["provenance/source-artifacts.json"]=(json.dumps(provenance,indent=2)+"\n").encode()
    files["provenance/hashes.sha256"]=("\n".join(f"{sha(files[p])}  {p}" for p in sorted(files))+"\n").encode()
    return files,summary

def materialize(dst):
    files,summary=archive()
    if dst.exists(): shutil.rmtree(dst)
    for rel,b in files.items():
        p=dst/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)
    return summary

def check():
    expected,summary=archive()
    got={p.relative_to(DATASET).as_posix():p.read_bytes() for p in DATASET.rglob("*") if p.is_file()} if DATASET.exists() else {}
    missing=sorted(set(expected)-set(got)); extra=sorted(set(got)-set(expected)); bad=sorted(k for k in expected.keys()&got.keys() if expected[k]!=got[k])
    if missing or extra or bad: raise SystemExit(f"corpus mismatch missing={missing} extra={extra} changed={bad}")
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("materialize","check")); ap.add_argument("--out-root",type=Path); a=ap.parse_args()
    result=materialize(a.out_root or DATASET) if a.command=="materialize" else check()
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
