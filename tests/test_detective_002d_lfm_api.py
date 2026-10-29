"""Offline DETECTIVE-002D contract and evidence tests."""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

from scripts import detective_002b_runtime as b
from scripts import detective_002d_lfm_api as d
from scripts import detective_002d_verify as v


class LfmApiTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.candidate,self.battery=d.load_contract()

    def test_only_api_handling_changes_from_002c_runtime(self):
        prior=json.loads(d.PRIOR_CONFIG.read_text())
        self.assertEqual(self.cfg["runtime"],prior["runtime"])
        self.assertEqual(
            self.cfg["model"]["weight_sha256"],
            prior["model"]["weight_sha256"])
        self.assertEqual(self.cfg["runtime"]["transformers"],"5.20.1")
        self.assertEqual(
            self.cfg["runtime"]["transformers_wheel_sha256"],
            "5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19")

    def test_prior_002c_failure_is_exact_and_weightless(self):
        p=json.loads(d.PRIOR_RESULT.read_text())
        self.assertEqual(p["status"],"tokenizer_incompatible")
        self.assertEqual(p["tokenizer_stage"]["tokenizer_class"],
                         "TokenizersBackend")
        self.assertEqual(p["tokenizer_stage"]["tokenizer_backend"],
                         "tokenizers")
        self.assertEqual(p["tokenizer_stage"]["tokenizer_error"]["type"],
                         "AttributeError")
        self.assertEqual(p["tokenizer_stage"]["tokenizer_error"]["message"],"")
        self.assertFalse(p["tokenizer_stage"]["source"]["weight_downloaded"])
        self.assertIsNone(p["model_stage"])

    def test_diagnostic_explicitly_requests_text_then_dict(self):
        q=self.cfg["diagnostic"]
        self.assertEqual(q["text_render"],{
            "tokenize":False,"add_generation_prompt":True,
            "return_dict":False
        })
        self.assertEqual(q["tokenized_render"],{
            "tokenize":True,"add_generation_prompt":True,
            "return_dict":True,"return_tensors":"pt"
        })
        self.assertTrue(q["download_weight_only_after_both_render_paths_pass"])
        self.assertEqual(q["require_mapping_keys"],["input_ids"])

    def test_source_blobs_and_license_remain_frozen(self):
        a=self.cfg["source_authority"]
        self.assertEqual(d.git_blob_sha1(d.SOURCE),a["stage_a_git_blob_sha1"])
        self.assertEqual(d.git_blob_sha1(d.BATTERY),a["battery_git_blob_sha1"])
        self.assertEqual(d.git_blob_sha1(d.PRIOR_CONFIG),
                         a["prior_config_git_blob_sha1"])
        self.assertEqual(d.git_blob_sha1(d.PRIOR_RESULT),
                         a["prior_result_git_blob_sha1"])
        self.assertTrue(self.cfg["model"]["production_license_gate"])
        self.assertFalse(self.cfg["decisions"]["production_license_cleared"])

    def test_zero_training_selection_relevance_and_expansion(self):
        self.assertEqual(self.cfg["runtime"]["optimizer_steps"],0)
        self.assertEqual(self.cfg["runtime"]["paid_api_calls"],0)
        self.assertFalse(self.cfg["decisions"]["training_selection_allowed"])
        self.assertFalse(self.cfg["decisions"]["population_expansion_admitted"])
        self.assertFalse(self.cfg["decisions"]["relevance_mapping_accessed"])

    def base_result(self,status="template_render_incompatible"):
        return {
            "schema":"detective-002d-lfm-api-result-v1",
            "experiment":"DETECTIVE-002D","issue":135,
            "status":status,
            "config_sha256":d.file_sha(d.CONFIG),
            "source_evidence_sha256":d.file_sha(d.SOURCE),
            "battery_sha256":d.file_sha(d.BATTERY),
            "prior_config_sha256":d.file_sha(d.PRIOR_CONFIG),
            "prior_result_sha256":d.file_sha(d.PRIOR_RESULT),
            "model":self.cfg["model"],
            "runtime_contract":self.cfg["runtime"],
            "resolved_packages":{
                "python":"3.12.15","torch":"2.14.1+cpu",
                "transformers":"5.20.1","huggingface_hub":"2.3.0",
                "tokenizers":"0.23.3","safetensors":"0.8.0"
            },
            "optimizer_steps":0,"paid_api_calls":0,
            "relevance_mapping_accessed":False,
            "production_license_cleared":False,
            "training_selection_allowed":False,
            "population_expansion_admitted":False,
            "diagnostic_stage":{
                "status":"template_render_incompatible",
                "source":{
                    "source_snapshot":"/tmp/source",
                    "config_sha256":self.cfg["model"]["config_sha256"],
                    "tokenizer_config_sha256":
                        self.cfg["model"]["tokenizer_config_sha256"],
                    "chat_template_sha256":
                        self.cfg["model"]["chat_template_sha256"],
                    "weight_downloaded":False
                },
                "transformers":"5.20.1",
                "tokenizer_class":"TokenizersBackend",
                "tokenizer_backend":"tokenizers",
                "tokenizer_load_seconds":0.1,
                "error":{
                    "phase":"text_render","task_id":"B01",
                    "type":"AttributeError","message":"","repr":"AttributeError()"
                },
                "prompts":[]
            },
            "model_stage":None,"wall_seconds":0.2,"peak_rss_kib":100000
        }

    def archive(self,root,result):
        rp=root/"result-source.json";d.write_json(rp,result)
        out=root/"archive"
        d.archive(rp,out,"a"*40,"123",
                  "https://github.com/techrote/neurounner/actions/runs/123")
        return out

    def test_template_failure_archive_proves_no_weight_or_model(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            out=self.archive(root,self.base_result())
            report=v.verify(out)
            self.assertEqual(report["status"],"template_render_incompatible")
            self.assertFalse(report["weight_downloaded"])

    def pass_result(self):
        r=self.base_result("runtime_compatibility_pass")
        prompts=[];encoded=[];tasks=[]
        for i,task in enumerate(self.battery["battery"]):
            messages=[
                {"role":"system","content":self.battery["system"]},
                {"role":"user","content":task["user"]}
            ]
            text=f"<formatted-{task['id']}>"
            tsha=__import__("hashlib").sha256(text.encode()).hexdigest()
            isha=(f"{i+1:x}"*64)[:64]
            prompts.append({
                "task_id":task["id"],"family":task["family"],
                "messages":messages,"formatted_text":text,
                "formatted_text_sha256":tsha
            })
            encoded.append({
                "task_id":task["id"],"family":task["family"],
                "mapping_type":"BatchEncoding",
                "mapping_keys":["attention_mask","input_ids"],
                "input_tokens":40+i,"input_ids_sha256":isha,
                "formatted_text_sha256":tsha
            })
            response={
                "B01":"Sensor trigger. Memory stores direction. Move.",
                "B02":"I remember 03:14, but that memory is not proof.",
                "B03":"The stamped record says 03:41, so I correct my account.",
                "B04":"I did not see the blue ceramic tag; I cannot confirm it."
            }[task["id"]]
            ids=[10+i,20+i]
            tasks.append({
                "task_id":task["id"],"family":task["family"],
                "messages":messages,"formatted_text_sha256":tsha,
                "prompt_token_count":40+i,"prompt_token_sha256":isha,
                "mapping_keys":["attention_mask","input_ids"],
                "prefill_seconds":0.1,"prompt_tokens_per_second":400.0,
                "generation_seconds":0.1,
                "generated_token_ids":ids,"generated_tokens":2,
                "completion_tokens_per_second":20.0,
                "hit_max_new_tokens":False,
                "response":response,"response_with_special_tokens":response,
                "observations":b.marker_observations(task,response)
            })
        r["diagnostic_stage"]={
            "status":"tokenizer_api_pass",
            "source":{
                "source_snapshot":"/tmp/source",
                "config_sha256":self.cfg["model"]["config_sha256"],
                "tokenizer_config_sha256":
                    self.cfg["model"]["tokenizer_config_sha256"],
                "chat_template_sha256":
                    self.cfg["model"]["chat_template_sha256"],
                "weight_downloaded":True
            },
            "transformers":"5.20.1",
            "tokenizer_class":"TokenizersBackend",
            "tokenizer_backend":"tokenizers",
            "tokenizer_load_seconds":0.1,
            "prompts":prompts,"tokenized_prompts":encoded
        }
        r["weight_stage"]={
            "selected_weight_path":self.cfg["model"]["weight_path"],
            "selected_weight_sha256":self.cfg["model"]["weight_sha256"],
            "selected_weight_bytes":self.cfg["model"]["weight_bytes"],
            "staged_weight_sha256":self.cfg["model"]["weight_sha256"]
        }
        r["model_stage"]={
            "status":"runtime_compatibility_pass",
            "environment":{
                "transformers":"5.20.1","torch_threads":4,
                "cuda_available":False
            },
            "tasks":tasks,"total_generated_tokens":8
        }
        return r

    def test_runtime_pass_archive_replays_api_and_model_prompt_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            out=self.archive(root,self.pass_result())
            report=v.verify(out)
            self.assertEqual(report["status"],"runtime_compatibility_pass")
            self.assertTrue(report["weight_downloaded"])

    def test_prompt_digest_tamper_refuses_even_after_rehash(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            out=self.archive(root,self.pass_result())
            result=d.read_json(out/"result.json")
            result["model_stage"]["tasks"][0]["prompt_token_sha256"]="f"*64
            d.write_json(out/"result.json",result)
            manifest=d.read_json(out/"manifest.json")
            manifest["result_sha256"]=d.file_sha(out/"result.json")
            d.write_json(out/"manifest.json",manifest)
            inv=[]
            for p in sorted(x for x in out.rglob("*") if x.is_file()):
                rel=p.relative_to(out).as_posix()
                if rel!="provenance/hashes.sha256":
                    inv.append(f"{d.file_sha(p)}  {rel}")
            (out/"provenance/hashes.sha256").write_text(
                "\n".join(inv)+"\n")
            with self.assertRaisesRegex(
                v.VerifyError,"model prompt differs from diagnostic gate"):
                v.verify(out)


if __name__=="__main__":
    unittest.main()
