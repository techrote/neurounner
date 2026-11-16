"""Side-by-side untuned/candidate gallery from original verified native PNGs.

Every frozen DEVELOPMENT view is retained. Invalid raw predictions are shown
as text refusals without an image. No teacher image or Python raster is used.
The HTML embeds unchanged PNG bytes and remains browsable as one local file.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil

from native_replay import ARMS, digest, read_json, relative_file, verify_file, verify_packet, write_json


def escape(value) -> str:
    return html.escape(str(value), quote=True)


def json_details(value) -> str:
    return escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def render(evidence: Path, proof: dict) -> tuple[dict, str, dict]:
    """Deterministically render data; image sources must already pass native verification."""
    summary, arms, report = proof["summary"], proof["arms"], proof["native_report"]
    native_results = {row["id"]: row for row in report["cases"]}
    outcomes = {(row["arm"], row["id"]): row for row in summary["cases"]}
    groups, items, copies = {}, [], {}
    for arm in ARMS:
        for outcome in arms[arm]["outcomes"]:
            row = dict(outcomes[(arm, outcome["id"])])
            row["raw_text"] = outcome["raw_text"]
            row["model_id"] = arms[arm]["document"]["model_id"]
            row["model_revision"] = arms[arm]["document"]["model_revision"]
            row["checkpoint"] = arms[arm]["document"]["checkpoint"]
            row["source_sha"] = summary["source_sha"]
            row["runtime_sha256"] = summary["runtime"]["native_sha256"]
            row["compiler_sha256"] = summary["compiler_sha256"]
            row["evaluation_sha256"] = summary["evaluation_sha256"][arm]
            if row["native_success"]:
                result = native_results[row["native_id"]]
                if result.get("success") is not True or not row["accepted"]:
                    raise ValueError("Gallery cannot display a rejected/unverified model drawing")
                sources = {"png": (result["png"], result["image_sha256"], "images"),
                           "receipt": (result["receipt"], result["receipt_sha256"], "receipts"),
                           "readback": (result["raw_materials"], result["readback_sha256"], "readbacks")}
                row["native_files"] = {}
                for name, (filename, expected, folder) in sources.items():
                    source = verify_file(evidence / "native", filename, expected)
                    target = folder + "/" + source.name
                    if target in copies:
                        raise ValueError("Duplicate gallery evidence target")
                    copies[target] = (source, expected)
                    row["native_files"][name] = {"path": target, "sha256": expected}
                png = sources["png"]
                row["image_data"] = "data:image/png;base64," + base64.b64encode(
                    verify_file(evidence / "native", png[0], png[1]).read_bytes()).decode("ascii")
                row.update({key: result[key] for key in ("program_sha256", "operations_sha256", "affected_cells",
                                                          "acknowledged_cells", "filled_cells", "structure",
                                                          "deterministic_replay", "history_and_reopen_checked")})
            items.append(row)
            group = groups.setdefault(row["structural_group"], {"family": row["family"], "design": row["design"],
                                                               "prompt": row["prompt"], "seeds": {}})
            group["seeds"].setdefault(row["seed"], {})[arm] = row
    cards = []
    for group_id, group in groups.items():
        seeds = []
        for seed, pair in sorted(group["seeds"].items()):
            figures = []
            for arm in ARMS:
                row = pair[arm]
                title = "Untuned baseline" if arm == "baseline" else "Trained candidate · step " + str(row["checkpoint"]["global_step"])
                if row["native_success"]:
                    drawing = ('<a class="drawing" download="' + escape(row["native_id"]) + '.png" href="' + row["image_data"] + '">'
                               '<img width="96" height="96" alt="Native stored material readback for ' + escape(arm + " " + row["id"]) +
                               '" src="' + row["image_data"] + '"></a>')
                    status = str(row["filled_cells"]) + " occupied cells · " + str(row["affected_cells"]) + " changed cells · fresh-World replay verified"
                    links = " · ".join('<a href="' + escape(item["path"]) + '" download>' + escape(name) + '</a>'
                                       for name, item in row["native_files"].items())
                    status += "<br>" + links
                else:
                    label = "Raw output rejected" if not row["accepted"] else "Native verification failed"
                    reason = row["refusal"] if not row["accepted"] else "; ".join(row["native_failures"])
                    drawing = '<div class="refusal"><strong>' + escape(label) + '</strong><p>' + escape(reason) + '</p><span>No model image</span></div>'
                    status = "No native drawing is credited to this view."
                agreement = ("Requested family matched" if row["family_agreement"] else "Requested family did not match") if row["accepted"] else "No accepted family proposal"
                raw_label = "Original raw output" if row["generation_attempted"] else "Unattempted generation and refusal"
                metadata = {key: value for key, value in row.items() if key not in ("image_data", "raw_text", "prompt")}
                figures.append('<figure><h4>' + escape(title) + '</h4>' + drawing + '<figcaption><p>' + status +
                               '</p><p class="family-status">' + escape(agreement) + '</p><details><summary>' + raw_label +
                               '</summary><pre>' + escape(row["raw_text"] or "[No output tokens]") + '</pre></details>' +
                               '<details><summary>Source, model, compiler and native receipts</summary><pre>' + json_details(metadata) +
                               '</pre></details></figcaption></figure>')
            seeds.append('<section class="seed"><h3>Host seed ' + str(seed) + '</h3><div class="pair">' + "".join(figures) + '</div></section>')
        cards.append('<article data-family="' + escape(group["family"]) + '"><div class="eyebrow">' + escape(group["family"]) +
                     ' · DEVELOPMENT</div><h2>' + escape(group["design"].replace("_", " ")) + '</h2><p class="request">' +
                     escape(group["prompt"]) + '</p><p class="group">' + escape(group_id) + '</p>' + "".join(seeds) + '</article>')
    metrics = []
    for arm in ARMS:
        stats = summary["arms"][arm]
        checkpoint = arms[arm]["document"]["checkpoint"]
        label = "Untuned baseline" if arm == "baseline" else "Candidate · " + str(checkpoint["global_step"]) + " optimizer steps"
        metrics.append('<div class="metric"><h2>' + escape(label) + '</h2><strong>' + str(stats["native_verified_views"]) +
                       '/12</strong><span> native-verified seed views</span><p>' + str(stats["actual_generations"]) +
                       '/6 actual generations · ' + str(stats["accepted_program_views"]) + '/12 strict accepted program views · ' +
                       str(stats["family_agreement_views"]) + '/12 accepted views matching the requested family</p></div>')
    failures = ""
    if summary["native_failures"]:
        failures = '<details open class="failures"><summary>Preserved native campaign failures</summary><pre>' + json_details(summary["native_failures"]) + '</pre></details>'
    status = "Native protocol verified" if summary["native_passed"] else "Native protocol incomplete / failed"
    if summary["counts"]["native_verified_views"] == 0:
        status += "; zero model drawings"
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CyberSand · Smol baseline and candidate</title><style>
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#0d151c;color:#e8e2cd}body{margin:0}
header,main,footer{max-width:1200px;margin:auto;padding:28px}header{border-bottom:1px solid #354957}
h1{font-size:clamp(32px,5vw,54px);line-height:1.08;margin:15px 0 22px}h2{font-size:23px;margin:7px 0}h3{font:14px ui-monospace,monospace;margin:25px 0 12px}h4{margin:0;padding:14px;font-size:17px}
p{color:#b5c5ce}.eyebrow{font:12px ui-monospace,monospace;color:#7bd1bd;text-transform:uppercase;letter-spacing:.08em}.identity,.group{font:12px/1.65 ui-monospace,monospace;overflow-wrap:anywhere}
.metrics,.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.metrics{margin:26px 0}.metric{padding:18px;border:1px solid #354957;background:#142530}.metric h2{font-size:17px}.metric strong{font-size:32px}.metric p{font-size:12px;margin:8px 0 0}
.toolbar{display:flex;gap:20px;align-items:center;margin-top:20px}select{font:inherit;color:#e8e2cd;background:#182732;border:1px solid #426074;padding:8px}
article{background:#101e28;border:1px solid #354957;padding:24px;margin-bottom:28px}.request{max-width:1000px}.group{color:#7895a5}
figure{margin:0;background:#0d151c;min-width:0;border:1px solid #263e4d}.drawing{display:block}img{display:block;width:100%;height:auto;aspect-ratio:1;image-rendering:pixelated}
figcaption{padding:14px}figcaption p{font-size:12px;margin:0 0 10px}.family-status{color:#8ed4c4}.refusal{min-height:210px;padding:25px;display:flex;flex-direction:column;justify-content:center;background:#211d20;color:#f4ad92}.refusal p{font:13px/1.6 ui-monospace,monospace;overflow-wrap:anywhere;color:#ddbab1}.refusal span{font-size:12px}
details{font-size:13px;margin:8px 0}summary{cursor:pointer}pre{font:11px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto}a{color:#7bd1bd}.failures{border:1px solid #c08063;padding:14px}.hidden{display:none}footer{border-top:1px solid #354957;font-size:13px}
@media(max-width:650px){header,main,footer{padding:16px}article{padding:12px}.pair{gap:8px}h4,figcaption{padding:10px}.refusal{padding:14px;min-height:250px}.metrics{grid-template-columns:1fr}}
</style></head><body><header><div class="eyebrow">CyberSand · SmolLM2-135M-Instruct · first DEVELOPMENT comparison</div>
<h1>Untuned baseline<br>and trained candidate</h1>
<p>Six structural requests across Greebles, Pipework and Architecture. Each request permits one first-pass generation per arm; accepted programs are compiled with two fixed host seeds, giving twelve recorded seed views per arm. Seed views are not independent model generations. The counters below state how many generations were actually attempted.</p>
<p>Only untouched, strict JSON/EOS outputs can enter native Art. Each displayed original PNG maps independently captured native material IDs to the declared palette. It is not a graphical Workbench screenshot. Raw refusals remain visible without replacement images.</p>
<p><strong>__STATUS__.</strong> Native execution and structural checks do not establish requested-layout fulfillment, artistic quality or human acceptance.</p>
<div class="metrics">__METRICS__</div><div class="identity">Model/native execution source: __SOURCE__<br>Compiler SHA-256: __COMPILER__<br>Native runtime SHA-256: __RUNTIME__<br>Original admitted corpus source: __CORPUS__<br>Godot: __GODOT__</div>
<div class="toolbar"><label>Assignment <select id="filter"><option value="">All families</option><option value="greebles">Greebles</option><option value="pipework">Pipework</option><option value="architecture">Architecture</option></select></label><span id="count"></span></div>__FAILURES__</header>
<main>__CARDS__</main><footer><p>Click an image to download its unchanged original PNG. Images, raw outputs and displayed metadata are embedded in this self-contained HTML. Receipt and readback download links require the extracted native archive with its gallery directory. The companion manifest binds each raw completion and model checkpoint to its compiler, native receipts, material readback and source hashes. Native Art retains material, focus, protection, Stop, shared Undo/Redo and persistence authority. Safety probes also run when every model output is rejected; that does not create model drawing evidence.</p>
<p>All rows are DEVELOPMENT. FINAL targets and images are excluded. Untuned versus candidate comparisons use the same frozen corpus, framing, numerical configuration and greedy decoding budget.</p></footer>
<script>const filter=document.getElementById('filter'),cards=[...document.querySelectorAll('article')];function update(){let count=0;for(const card of cards){const visible=!filter.value||card.dataset.family===filter.value;card.classList.toggle('hidden',!visible);if(visible)count++;}document.getElementById('count').textContent=count+' structures · two seeds · two arms';}filter.addEventListener('change',update);update();</script></body></html>"""
    replacements = {"__STATUS__": escape(status), "__METRICS__": "".join(metrics), "__SOURCE__": escape(summary["source_sha"]),
                    "__COMPILER__": escape(summary["compiler_sha256"]), "__RUNTIME__": escape(summary["runtime"]["native_sha256"]),
                    "__CORPUS__": escape(summary["corpus"]["source_sha"]), "__GODOT__": escape(summary["runtime"]["godot_version"]),
                    "__FAILURES__": failures, "__CARDS__": "".join(cards)}
    for token, value in replacements.items():
        page = page.replace(token, value)
    manifest = {"schema_version": 1, "kind": "smol-model-native-development-gallery",
                "model_native_summary_sha256": digest(evidence / "model-native-summary.json"),
                "source_sha": summary["source_sha"], "corpus_source_sha": summary["corpus"]["source_sha"],
                "runtime_sha256": summary["runtime"]["native_sha256"], "included_splits": ["DEVELOPMENT"],
                "counts": summary["counts"], "arms": summary["arms"], "native_passed": summary["native_passed"],
                "claim_limits": summary["claim_limits"],
                "items": [{key: value for key, value in row.items() if key not in ("image_data", "raw_text")} for row in items],
                "index_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest()}
    return manifest, page, copies


def verify_gallery(evidence: Path, output: Path, proof: dict) -> dict:
    manifest, page, copies = render(evidence, proof)
    if read_json(relative_file(output, "manifest.json")) != manifest:
        raise ValueError("Gallery manifest differs from the verified model/native packet")
    if relative_file(output, "index.html").read_bytes() != page.encode("utf-8"):
        raise ValueError("Gallery HTML differs from its deterministic verified rendering")
    for name, (_source, expected) in copies.items():
        verify_file(output, name, expected)
    return {"manifest_sha256": digest(output / "manifest.json"), "index_sha256": manifest["index_sha256"],
            "native_images": len(copies) // 3, "case_views": len(manifest["items"])}


def build(evidence: Path, output: Path) -> dict:
    proof = verify_packet(evidence, details=True)
    manifest, page, copies = render(evidence, proof)
    output.mkdir(parents=True, exist_ok=False)
    for name, (source, expected) in copies.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError("Original native gallery evidence changed while copying")
    (output / "index.html").write_text(page, encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    verify_gallery(evidence, output, proof)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.evidence, args.output)
    print(json.dumps({"source_sha": result["source_sha"], "counts": result["counts"], "output": str(args.output)}))
