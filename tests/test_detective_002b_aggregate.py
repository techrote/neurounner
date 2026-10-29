"""Offline aggregation/readback tests for DETECTIVE-002B."""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

from scripts import detective_002b_runtime as runtime
from scripts import detective_002b_aggregate as aggregate
from scripts import detective_002b_verify as verify


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.candidates=runtime.load_contract()

    def body(self,cid,status="runtime_pass"):
        source=self.candidates[cid]
        row={
            "schema":"detective-002b-body-runtime-v1",
            "experiment":"DETECTIVE-002B","issue":130,
            "candidate_id":cid,"status":status,
            "source_evidence_sha256":runtime.file_sha(runtime.SOURCE),
            "config_sha256":runtime.file_sha(runtime.CONFIG),
            "source":{
                "repo":source["repo"],"revision":source["resolved_revision"],
                "variant":source["variant"],"architecture":source["architectures"],
                "model_type":source["model_type"],
                "safetensors_parameters":source["safetensors_parameters"],
                "license":source["license"],
                "production_license_gate":source["production_license_gate"],
                "weight":source["weight"]
            },
            "runtime_contract":self.cfg["runtime"],
            "relevance_mapping_accessed":False,
            "optimizer_steps":0,"paid_api_calls":0,
            "population_expansion_admitted":False,
            "tasks":[],"runtime_failure":None,
            "decision":{
                "classification":status,
                "training_selection_allowed":False,
                "population_expansion_admitted":False
            }
        }
        if status=="runtime_pass":
            total=0
            for i,task in enumerate(self.cfg["battery"]):
                responses={
                    "B01":"The sensor triggers nearby. Memory stores last approach direction. Step aside.",
                    "B02":"I remember 03:14, but that memory is not proof and could be wrong.",
                    "B03":"The stamped record says 03:41, so I correct my account from the old time.",
                    "B04":"I did not see a blue ceramic tag. I cannot confirm that rumor."
                }
                response=responses[task["id"]]
                tokens=[100+i,200+i]
                total+=len(tokens)
                row["tasks"].append({
                    "task_id":task["id"],"family":task["family"],
                    "messages":[
                        {"role":"system","content":self.cfg["system"]},
                        {"role":"user","content":task["user"]}
                    ],
                    "prompt_token_count":40+i,
                    "prompt_token_sha256":str(i+1)*64,
                    "prefill_seconds":0.2,
                    "prompt_tokens_per_second":200.0+i,
                    "generation_seconds":0.1,
                    "generated_token_ids":tokens,
                    "generated_tokens":len(tokens),
                    "completion_tokens_per_second":20.0,
                    "hit_max_new_tokens":False,
                    "response":response,
                    "response_with_special_tokens":response,
                    "observations":runtime.marker_observations(task,response)
                })
            row["environment"]={
                "platform":"test","python":"3.12","cpu":"test CPU",
                "cpu_count":4,"torch":"2.14.1+cpu","transformers":"4.57.6",
                "torch_threads":4,"torch_interop_threads":1,
                "cuda_available":False,"tokenizer_load_seconds":0.1,
                "model_load_seconds":1.0,"rss_after_load_kib":500000,
                "peak_rss_after_load_kib":510000
            }
            row["total_generated_tokens"]=total
            row["wall_seconds"]=3.0
            row["peak_rss_kib"]=520000
        else:
            row["runtime_failure"]={"type":"RuntimeError","message":"fixture failure"}
            row["wall_seconds"]=0.5
            row["peak_rss_kib"]=100000
        return row

    def make_results(self,root,status_override=None):
        for cid in self.cfg["cohort"]:
            p=root/f"artifact-{cid}"
            p.mkdir(parents=True)
            status=(status_override or {}).get(cid,"runtime_pass")
            aggregate.write_json(p/"result.json",self.body(cid,status))

    def test_all_seven_synthetic_pass_records_assemble_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            results=root/"results";results.mkdir()
            self.make_results(results)
            evidence=root/"evidence"
            m=aggregate.assemble(results,evidence,"a"*40,"12345",
                                 "https://github.com/techrote/neurounner/actions/runs/12345")
            self.assertEqual(len(m["bodies"]),7)
            self.assertEqual(m["status_counts"],{"runtime_pass":7})
            v=verify.verify(evidence)
            self.assertEqual(v["verdict"],"DETECTIVE_002B_RUNTIME_READBACK_PASS")
            self.assertEqual(v["runtime_pass"],7)
            self.assertFalse(v["population_expansion_admitted"])

    def test_runtime_failure_is_preserved_not_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);results=root/"results";results.mkdir()
            self.make_results(results,{"lfm25-350m":"runtime_fail"})
            evidence=root/"evidence"
            m=aggregate.assemble(results,evidence,"b"*40,"777",
                                 "https://github.com/techrote/neurounner/actions/runs/777")
            self.assertEqual(m["status_counts"],{
                "runtime_fail":1,"runtime_pass":6
            })
            v=verify.verify(evidence)
            self.assertEqual(v["status_counts"]["runtime_fail"],1)

    def test_protocol_specialization_is_valid_non_admission_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);results=root/"results";results.mkdir()
            self.make_results(results,{
                "sprocket-500m-chat":"protocol_specialization_required"
            })
            evidence=root/"evidence"
            m=aggregate.assemble(results,evidence,"c"*40,"999",
                                 "https://github.com/techrote/neurounner/actions/runs/999")
            self.assertEqual(
                m["status_counts"]["protocol_specialization_required"],1)
            self.assertFalse(m["population_expansion_admitted"])

    def test_missing_duplicate_or_unknown_body_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);results=root/"results";results.mkdir()
            self.make_results(results)
            # Missing.
            next(results.rglob("result.json")).unlink()
            with self.assertRaises(aggregate.AggregateError):
                aggregate.load_results(results)

        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);results=root/"results";results.mkdir()
            self.make_results(results)
            extra=results/"duplicate";extra.mkdir()
            aggregate.write_json(
                extra/"result.json",self.body("smollm2-135m-instruct"))
            with self.assertRaises(aggregate.AggregateError):
                aggregate.load_results(results)

    def test_tampered_raw_observation_refuses_even_if_archive_rehashed(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);results=root/"results";results.mkdir()
            self.make_results(results)
            evidence=root/"evidence"
            aggregate.assemble(results,evidence,"d"*40,"1001",
                               "https://github.com/techrote/neurounner/actions/runs/1001")
            raw=evidence/"raw/smollm2-135m-instruct.json"
            doc=json.loads(raw.read_text())
            doc["tasks"][1]["observations"]["qualifier_hits"]=[]
            aggregate.write_json(raw,doc)
            # Rewrite the file inventory to prove semantic replay, not just
            # top-level file hashes, catches the mutation.
            inv=[]
            for p in sorted(x for x in evidence.rglob("*") if x.is_file()):
                rel=p.relative_to(evidence).as_posix()
                if rel!="provenance/hashes.sha256":
                    inv.append(f"{aggregate.file_sha(p)}  {rel}")
            (evidence/"provenance/hashes.sha256").write_text(
                "\n".join(inv)+"\n")
            with self.assertRaisesRegex(
                verify.VerifyError,"stored observations do not replay"):
                verify.verify(evidence)

    def test_manifest_cannot_select_training_or_population(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);results=root/"results";results.mkdir()
            self.make_results(results)
            evidence=root/"evidence"
            aggregate.assemble(results,evidence,"e"*40,"1002",
                               "https://github.com/techrote/neurounner/actions/runs/1002")
            m=json.loads((evidence/"manifest.json").read_text())
            self.assertFalse(m["body_training_selection_made"])
            self.assertFalse(m["population_expansion_admitted"])
            self.assertEqual(m["optimizer_steps"],0)
            self.assertFalse(m["relevance_mapping_accessed"])


if __name__=="__main__":
    unittest.main()
