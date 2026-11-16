"""D-2 matched synthetic-control comparison. No native/model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab import VERSION, packed, sha, simulate, raster


def score_weathering(result, recipe):
    """Measure actual erased pixels and cursor offset to frozen ORIGINAL contour."""
    original=result["before"]
    final=result["after"]
    erased=original-final
    edge={p for a,b in zip(result["geometry"]["original_perimeter"],
                            result["geometry"]["original_perimeter"][1:])
          for p in raster(a,b)}
    steps=[e for e in result["records"] if e["kind"]=="STEP"]
    target=recipe.get("boundary_offset",2)
    deviations=[abs(min(abs(e["point"][0]-x)+abs(e["point"][1]-y)
                        for x,y in edge)-target) for e in steps]
    return dict(strokes=len(steps),distinct_erased_cells=len(erased),
                erase_per_stroke=(len(erased)/len(steps) if steps else 0),
                original_rock_retained=len(final & original),
                boundary_target_offset=target,
                mean_cursor_offset_error=(sum(deviations)/len(deviations)
                                          if deviations else None),
                brush_up_reversals=sum(e["kind"]=="REVERSAL" for e in result["records"]),
                trapped=sum(e["kind"]=="TRAPPED" for e in result["records"]))


def compare_weathering(recipe):
    """Matched source, World, seed, step budget and permissions, NO model arms."""
    assert recipe["version"]==VERSION and recipe["family"]=="weathering_worm"
    policies=("boundary_biased","random_legal")
    arms={p:simulate(dict(recipe,walker_policy=p)) for p in policies}
    left,right=(arms[p] for p in policies)
    assert left["before"]==right["before"]
    assert left["guidance"]==right["guidance"]
    assert left["geometry"]==right["geometry"]
    def decisions(r):
        return [e for e in r["records"] if e["kind"]=="SYNTHETIC_CHOICE"]
    a,b=decisions(left),decisions(right)
    if a and b:
        assert a[0]["legal_catalog_sha256"]==b[0]["legal_catalog_sha256"]
    first_choice=next((i for i,(x,y) in enumerate(zip(a,b))
                       if x["chosen"]!=y["chosen"]),None)
    xsteps=[e for e in left["records"] if e["kind"]=="STEP"]
    ysteps=[e for e in right["records"] if e["kind"]=="STEP"]
    first_material=next((i for i,(x,y) in enumerate(zip(xsteps,ysteps))
                         if x["material_after_sha256"]!=y["material_after_sha256"]),None)
    return dict(version="d2.synthetic-weathering-pair/1",
                qualification="DEVELOPMENT_SYNTHETIC_ONLY",
                native=None,model=None,native_replay_verified=False,
                base_recipe_sha256=sha(packed(recipe)),
                matched_initial_material_sha256=sha(packed(sorted(map(list,left["before"])))),
                matched_guidance_sha256=sha(packed(sorted([list(p)+[v]
                                                   for p,v in left["guidance"].items()]))),
                matched_geometry_sha256=sha(packed(left["geometry"])),
                first_choice_divergence_index=first_choice,
                first_synthetic_material_divergence_step=first_material,
                comparisons={p:dict(metrics=score_weathering(arms[p],recipe),
                                    final_material_sha256=arms[p]["summary"]["material_sha256"],
                                    trajectory_sha256=arms[p]["summary"]["trajectory_sha256"],
                                    synthetic_steps=arms[p]["summary"]["steps"],
                                    original_event_sha256=arms[p]["summary"]["event_sha256"])
                             for p in policies})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--recipe",required=True,help="Existing DEVELOPMENT recipe JSON")
    parser.add_argument("--out",required=True,help="File to write result JSON")
    options=parser.parse_args()
    recipe=json.loads(Path(options.recipe).read_text())
    result=packed(compare_weathering(recipe))
    path=Path(options.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(result)
    print(json.dumps(dict(output=str(path),sha256=sha(result),
                          qualification="DEVELOPMENT_SYNTHETIC_ONLY"),sort_keys=True))
