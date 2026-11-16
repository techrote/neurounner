# DG007 — learned game-action receiver (Draft/HOLD)

DG005/DG006 established supervised source-state decoding and learned atomic state editors, but all their reported game actions came from a scripted projection. DG007 trains a separate game-action policy, then evaluates it using frozen source-derived DG006 law states on unseen synthetic game worlds.

Two actual CPU models per seed: law-conditioned 6-action candidate scorer and same-capacity law-blind control. 4,800 synthetic 9×9 material/obstacle/protected TRAIN worlds, 1,000 VALID worlds, 1,800 sealed worlds; two independent seeds. Host-masked LEFT/RIGHT/UP/DOWN/PAINT/RETRACE and 22-decision synthetic rollouts. Scripted expert labels train the policy, not the donor, and policy must learn game choices from three law bits.

Donor inputs: original DG002 Qwen2.5-0.5B attention-V, supervised DG005 two-scalar decoder (frozen), DG006 contextual/additive state editors (frozen), third material/protection bit externally supplied. Compare four/eight unseen edit sequences under real/zero/shuffled/oracle source, additive comparator and identical-world paired counterfactual law edits. Report raw choices, trained .pt checkpoints, hashes, PNGs and negative findings as Actions artifacts. A high score demonstrates only synthetic imitation, NOT discovered physics, native cross-model KV transplant, or genuine CyberSand gameplay. Previously inspected source families do NOT count as independent held-out source validation.

Hard boundary: Never modify CyberSand native/Godot/Workbench/map/material/asset/game code, production settings or production providers. Neurounner model experiments only. No paid API/GPU. Draft/HOLD, never merge or mark Ready.

Reproduce: run python -m experiments.memplant.learned_game_007.experiment --selftest; restore original DG002 source run 43736088404, DG005 decoder run 43795069697, DG006 models run 43798092630. Use --source, --reference-dg005, --reference-dg006, --out, --seed, --updates 500.
