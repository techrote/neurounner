"""Synthetic archive/readback tests for DETECTIVE-002C."""
from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest

from scripts import detective_002b_runtime as b
from scripts import detective_002c_lfm_runtime as r
from scripts import detective_002c_verify as v


class LfmV5VerifyTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.candidate,self.battery=r.load_contract()

    def base(self):
        return {
            "schema":"detective-002c-lfm-runtime-result-v1",
            "experiment":"DETECTIVE-002C","issue":133,
            "status":"tokenizer_incompatible",
            "config_sha256":r.file_sha(r.CONFIG),
            "source_evidence_sha256":r.file_sha(r.SOURCE),
            "battery_sha256":r.file_sha(r.BATTERY),
            "historical_result_sha256":r.file_sha(r.HISTORICAL),
            "model":self.cfg["model"],
            "runtime_contract":self.cfg["runtime"],
            "historical_control":self.cfg["historical_control"],
            "resolved_packages":{
                "python":"3.12.15","torch":"2.14.1+cpu",
                "transformers":"5.20.1","huggingface_hub":"1.0.0",
                "tokenizers":"0.22.0","safetensors":"0.8.0",
            },
            "relevance_mapping_accessed":False,
            "optimizer_steps":0,"paid_api_calls":0,
            "production_license_cleared":False,
            "training_selection_allowed":False,
            "population_expansion_admitted":False,
            "tokenizer_stage":{
                "status":"tokenizer_incompatible",
                "source":{
                    "source_snapshot":"/tmp/source",
                    "config_sha256":self.cfg["model"]["config_sha256"],
                    "tokenizer_config_sha256":
                        self.cfg["model"]["tokenizer_config_sha256"],
                    "chat_template_sha256":
                        self.cfg["model"]["chat_template_sha256"],
                    "weight_downloaded":False,
                },
                "transformers":"5.20.1",
                "tokenizer_error":{
                    "type":"ValueError","message":"synthetic tokenizer failure"},
                "tokenizer_load_seconds":0.1,
                "rendered_prompts":[],
            },
            "model_stage":None,
            "wall_seconds":0.2,"peak_rss_kib":100000,
        }

    def archive(self,root,result):
        result_path=root/"result-source.json"
        r.write_json(result_path,result)
        archive=root/"archive"
        r.archive(
            result_path,archive,"a"*40,"12345",
            "https://github.com/techrote/neurounner/actions/runs/12345")
        return archive

    def test_tokenizer_failure_proves_weight_was_not_downloaded(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            archive=self.archive(root,self.base())
            report=v.verify_archive(archive)
            self.assertEqual(report["status"],"tokenizer_incompatible")
            self.assertFalse(report["weight_downloaded"])
            self.assertEqual(report["optimizer_steps"],0)
            self.assertFalse(report["production_license_cleared"])

    def pass_result(self):
        d=self.base()
        d["status"]="runtime_compatibility_pass"
        rendered=[]
        tasks=[]
        for i,task in enumerate(self.battery["battery"]):
            response={
                "B01":"Sensor trigger. Memory stores direction. Move aside.",
                "B02":"I remember 03:14, but that memory is not proof.",
                "B03":"The stamped record says 03:41, so I correct my account.",
                "B04":"I did not see a blue ceramic tag; I cannot confirm it.",
            }[task["id"]]
            digest=(str(i+1)*64)[:64]
            messages=[
                {"role":"system","content":self.battery["system"]},
                {"role":"user","content":task["user"]},
            ]
            rendered.append({
                "task_id":task["id"],"family":task["family"],
                "messages":messages,"prompt_tokens":40+i,
                "prompt_token_sha256":digest,
            })
            tokens=[10+i,20+i]
            tasks.append({
                "task_id":task["id"],"family":task["family"],
                "messages":messages,
                "prompt_token_count":40+i,
                "prompt_token_sha256":digest,
                "prefill_seconds":0.1,
                "prompt_tokens_per_second":400.0,
                "generation_seconds":0.1,
                "generated_token_ids":tokens,
                "generated_tokens":len(tokens),
                "completion_tokens_per_second":20.0,
                "hit_max_new_tokens":False,
                "response":response,
                "response_with_special_tokens":response,
                "observations":b.marker_observations(task,response),
            })
        d["tokenizer_stage"]={
            "status":"tokenizer_pass",
            "source":{
                "source_snapshot":"/tmp/source",
                "config_sha256":self.cfg["model"]["config_sha256"],
                "tokenizer_config_sha256":
                    self.cfg["model"]["tokenizer_config_sha256"],
                "chat_template_sha256":
                    self.cfg["model"]["chat_template_sha256"],
                "weight_downloaded":True,
            },
            "transformers":"5.20.1",
            "tokenizer_class":"TokenizersBackend",
            "tokenizer_backend":"tokenizers",
            "tokenizer_load_seconds":0.1,
            "rendered_prompts":rendered,
        }
        d["weight_stage"]={
            "selected_weight_path":self.cfg["model"]["weight_path"],
            "selected_weight_sha256":self.cfg["model"]["weight_sha256"],
            "selected_weight_bytes":self.cfg["model"]["weight_bytes"],
            "staged_weight_sha256":self.cfg["model"]["weight_sha256"],
        }
        d["model_stage"]={
            "status":"runtime_compatibility_pass",
            "environment":{
                "transformers":"5.20.1","torch_threads":4,
                "cuda_available":False,
            },
            "tasks":tasks,
            "total_generated_tokens":8,
        }
        return d

    def test_pass_archive_replays_prompts_observations_and_weight(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            archive=self.archive(root,self.pass_result())
            report=v.verify_archive(archive)
            self.assertEqual(report["status"],"runtime_compatibility_pass")
            self.assertTrue(report["weight_downloaded"])

    def test_tokenizer_failure_with_weight_stage_refuses_even_if_rehashed(self):
        d=self.base()
        d["weight_stage"]={
            "selected_weight_path":"model.safetensors",
            "selected_weight_sha256":self.cfg["model"]["weight_sha256"],
        }
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            archive=self.archive(root,d)
            with self.assertRaisesRegex(
                v.VerifyError,"improperly entered model stage"):
                v.verify_archive(archive)

    def test_tampered_observation_refuses_after_archive_rehash(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            archive=self.archive(root,self.pass_result())
            result=r.read_json(archive/"result.json")
            result["model_stage"]["tasks"][1]["observations"]["qualifier_hits"]=[]
            r.write_json(archive/"result.json",result)
            manifest=r.read_json(archive/"manifest.json")
            manifest["result_sha256"]=r.file_sha(archive/"result.json")
            r.write_json(archive/"manifest.json",manifest)
            lines=[]
            for p in sorted(x for x in archive.rglob("*") if x.is_file()):
                rel=p.relative_to(archive).as_posix()
                if rel!="provenance/hashes.sha256":
                    lines.append(f"{r.file_sha(p)}  {rel}")
            (archive/"provenance/hashes.sha256").write_text(
                "\n".join(lines)+"\n")
            with self.assertRaisesRegex(
                v.VerifyError,"mechanical observations do not replay"):
                v.verify_archive(archive)


if __name__=="__main__":
    unittest.main()
