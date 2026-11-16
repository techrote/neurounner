# MUTAGEN-M1 — Longitudinal d1 worm personality comparison

[Owner-authorised follow-on to #620](https://github.com/techrote/cybersand/issues/620) and [M0 original cloud evidence in Draft #622](https://github.com/techrote/cybersand/pull/622). **This is a new separate Draft/HOLD source branch** stacked on #622. No modifications to accepted native CyberSand, stock d1 or another worker's source branch.

## Goal

Verify whether crude weight edits produce persistent *recognisable* learned steering habits over longer sequences and a second world, instead of short eight-decision quirks. The question is artistic/behavioural diversity, not superior A* navigation.

| Experimental arm | Deliberate weight mutation | Worlds and themes | Decision budget per episode |
|---|---|---|---|
| Original stock d1 | None; exact Q8 original | W0 seeds 570 and 2111 × Greebles/Pipework/Architecture | 128 |
| MLP Scoop | M0 identical 25% of chosen middle FFN values ×0.25 | Same | 128 |
| Attention Gouge | M0 identical 25% of selected global Q/K/output signs inverted | Same | 128 |
| Conv Scrape 10 | Same six convolution-layer tensor groups as M0 but zero only first 10% | Same | 64 |
| Conv Scrape 25 | Same groups, first 25% zero | Same | 64 |

Original convicted 50% Conv Scrape all-LEFT model remains preserved in [M0 original evidence](https://github.com/techrote/cybersand/issues/620#issuecomment-6900453650); no point in repeating its failures without a new question.

**Two separate ordinary hosted GitHub CPU VMs maximum at once; one resident d1 model per VM.** One GitHub Actions matrix run performs these five model jobs, each with bounded deadlines and always-upload original artifacts. Later a read-only comparison job checks matched initial observations and builds a grid from original W0-generated PNGs. The stock control is run ONCE as a separate arm rather than rebuilt for every mutant.

## Method

- Exact pinned `LiquidAI/d1-2B-GGUF` Q8 donor (SHA256 `073313d9...`), pinned System One llama.cpp `b11960`, pinned W0 original `4bb9d6e...` (Git blob `8979b6f...`). Source identities and original full hashes are checked in workflow and each model-run evidence.
- For MLP and Attention, compare freshly carved model SHA against immutable successful M0 mutation bytes before inference. For Conv10/25 inspect actual named `shortconv.in_proj/out_proj/conv` GGUF tensors, retain unique SHA and exact changed counts.
- Per tick the real d1 `/v1/systemone` model chooses from W0 host-legal actions, with original option probabilities, input observations, CPU decision timing and failure responses recorded. No rule-based shadow policy masquerading as model output; no illegal coordinate authority.
- Each genuine STOP ends the episode normally, even if it occurs before 64/128. Any failure preserves raw decisions and an explicitly labelled partial W0 replay.
- Same two world seeds, same initial geometry, same three themes, same prompt semantics and legal-action interface across arms. After the first model decision the worlds naturally diverge; **matched starting conditions** do not mean identical later observations.
- Original PNGs are **source-faithful Python W0 material-trail renderings, NOT actual native Workbench/Godot PAINT**. No model optimisation or native edit. Small 64/128 cohorts are exploratory, not a claim of generalisation.
- Mean/median/p95 model-only latency is measured, but no speed advantage is expected from Q8 code modifications that leave all matrix operations in place.

## Source/data boundary and status

Do not merge, mark Ready, enable auto-merge, launch a recurring worker, alter #622 or #611, or promote production providers. The explicit `longrun-request.json` is the **sole push trigger** for substantial model runs. Before creating it, cheap PR/source tests must pass. Preserve all failed and successful run artifacts, including raw attention/FFN/convolution mutations, model decisions and comparative visuals.

The comparative result should report per-seed action histograms, revisits, branches, early STOP rate, 128/64-step summaries, speed costs and representative original galleries; if the model collapses back to one action under longer play, say so without hiding it.
