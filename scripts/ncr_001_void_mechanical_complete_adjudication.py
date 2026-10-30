#!/usr/bin/env python3
"""Source-only final mechanical-coverage reconciliation for void.1 #72."""
from __future__ import annotations
import argparse,json,hashlib,subprocess
from pathlib import Path
import ncr_001_void_poststep2048_adjudication as prior
import ncr_001_void_poststep2048_masked_kv_original_custody as custody
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"experiments/ncr-001/adjudication/void1-poststep2048-scientific-boundary-20261030.json"
OUTPUT=ROOT/"experiments/ncr-001/adjudication/void1-mechanical-coverage-complete-20261030.json"
PRIOR_BLOB="27b28cc7c0d878e4999458331c17340ec6037531"
MANIFEST_BLOB="439584bd144c5b640cbc49807200ea3845b384cb"
STATUS="VOID1_LOCKED_72_MECHANICAL_EVIDENCE_COVERAGE_COMPLETE__FINAL_SCIENTIFIC_ADMISSION_PENDING"
def refuse(ok,why):
 if not ok: raise ValueError("VOID_FINAL_MECH_REFUSED: "+why)
def blob(raw): return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def pin(path,sha):
 refuse(path.is_file() and not path.is_symlink() and blob(path.read_bytes())==sha,"blob "+str(path))
 rel=path.relative_to(ROOT).as_posix()
 x=subprocess.run(["git","-C",str(ROOT),"ls-tree","HEAD","--",rel],capture_output=True,text=True)
 refuse(x.stdout==f"100644 blob {sha}\t{rel}\n","tree "+rel)
def build():
 pin(PRIOR,PRIOR_BLOB);pin(custody.MANIFEST,MANIFEST_BLOB)
 old=prior.build_report();ref=json.loads(PRIOR.read_text());refuse(old==ref,"#219 decision drift")
 new=custody.audit(live=False);p=new["physical_result"]
 refuse(new["status"]=="PASS_ORIGINAL_POSTSTEP_B2_MASKED_KV_BYTES__SCIENTIFIC_72_UNADMITTED"
        and new["permanent_git_custody"] and not new["train_authorized"],"custody boundary")
 for side in ("left","right"):
  x=p["sides"][side]
  refuse(x["width"]==2048 and x["genuine_B2_incremental_steps"]==64
         and x["tail_real_incremental_units_per_row"]==64
         and x["full_vs_cached_max_abs"]<=5e-4
         and x["batch_vs_single_cached_max_abs"]<=5e-4
         and x["row_swap_cached_max_abs"]<=5e-4
         and x["explicit_attention_mask"] and x["explicit_mask_derived_position_ids"],
         "masked cached evidence "+side)
 refuse(old["all_nine_have_bounded_source_bound_physical_observations"]
        and old["largest_prior_joint_gap_poststep_2048"]=="CLOSED_BY_COHORT_E"
        and old["masked_B2_incremental_cache_measured"] is False
        and old["per_model_conformance_pass"] is False,"prior scientific boundary")
 remaining=[
  "explicit_final_model_specific_scientific_72_admission_not_granted",
  "fresh_independent_host_replication_not_measured",
  "multi_sample_multi_seed_learned_effectiveness_not_measured",
  "matched_NCA_MLP_training_quality_and_compute_not_measured",
  "generated_EOS_258_skill_not_qualified",
  "historical_TRAIN002b_EOS_effect_remains_unmeasured"]
 return {"schema":"ncr001-void1-mechanical-complete-v1","status":STATUS,
  "parent_pr":221,"parent_head":"d48b005cefbb95e22be9dc45c4fc5de819a4c448",
  "prior_219_status":old["status"],"historical_171_adjudication_unchanged":old["historical_171_adjudication_unchanged"],
  "new_masked_cache_original_run":41136709967,"new_masked_cache_source_head":"b3e9e6a05e007297de19f3955b59e2a9eb118996",
  "masked_B2_incremental_cache_measured":True,"masked_B2_incremental_tail_units_per_row":64,
  "masked_B2_left_full_vs_cached_max_abs":p["sides"]["left"]["full_vs_cached_max_abs"],
  "masked_B2_right_full_vs_cached_max_abs":p["sides"]["right"]["full_vs_cached_max_abs"],
  "all_nine_locked_72_check_names_have_bounded_physical_evidence":True,
  "expanded_cache_batch_mask_generation_mechanical_coverage_present":True,
  "known_locked_72_mechanical_evidence_gap":None,
  "remaining_scientific_or_downstream_gaps":remaining,
  "remaining_items_not_retroactive_new_72_mechanical_checks":remaining[1:],
  "source_only_adjudication":True,"fresh_physical_execution_in_this_adjudication":False,
  "new_optimizer_steps_in_this_adjudication":0,"heldouts_opened":False,
  "independent_scientific_72_admission":False,"per_model_conformance_pass":False,
  "full_72_conformance_pass":False,"actual_compute_budget_qualified":False,
  "train_authorized":False,
  "decision_note":"The exact void.1 evidence graph now covers the locked #72 mechanical checks plus expanded long-context/cache/batch/mask/generation mechanics. This same-account source adjudication does not itself grant the explicit final scientific per-model admission."}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path);ap.add_argument("--check",action="store_true");a=ap.parse_args();r=build()
 if a.check: refuse(OUTPUT.is_file() and json.loads(OUTPUT.read_text())==r,"committed decision mismatch")
 if a.out: a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
 print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
