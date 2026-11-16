# DRIV3R D020 — branch-retention constrained student

16 November 2026 · **research-only / negative / do not promote**.

Three 4,714-parameter standalone students completed **480 Adam updates** (160 each) from a 60,366-state archived D018 synthetic teacher corpus. Positive and negative branch-margin supervision was paired with host-legal teacher KL and a hard validation rule: normal/trap branching no more than 1 below D017; normal/trap coverage no more than 2 cells below D017; squeeze coverage no more than 1.5 below; zero illegal actions. No test data entered training or selection.

**Selection:** D020_low passes validation. On previously unseen test seeds 53001–53016, it increases normal branches by +1.0312, trap branches by +0.5312, and trap cells by +0.8125 versus D017. On independent confirmation seeds 54001–54016, trap coverage **falls by −2.5625 cells** (seed-block 95% bootstrap interval **[−4.7969, −0.2496]**), violating the predeclared hard threshold. No illegal actions. **REJECT D020**; D007 remains gameplay champion and D017 is the supported auxiliary branching adapter.

The laboratory's original source in this PR is `branch_retention_020.py` (Git blob `15da11ab4f93154e4f7715f154941fd2bacdee28`) and `test_branch_retention_020.py` (Git blob `d28daa0202c2f6333e780421aa762e78cf14af5f`). Three new unit tests passed. Independent locally reconstructed synthetic-host replay reproduced 960 test episodes, 960 confirmation episodes, 3,866 action receipts, 60 PNGs byte-exact, and all summary and bootstrap metrics. Input/source snapshots were SHA-256-pinned:

- D007 `1a4a35f8f15f979b8460121d31021d562a457de78d86de9b9d2fedbcec05e38f`
- D017 `715dff35c08a4d19bb7f523925eace020f35bdcaa3bc7bfe44bf2aa67606f11f`
- Original source Q8 `140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522`
- D018 compact start `35133e880dbf27affec4aebb2859d7052c13aa53f0f8c953c3931a855618ad21`
- D019b compact start `6f736eb56bc01c2e313e6f194a81f08c878d24261a24db8ff3d3749c027174ec`
- D018 synthetic corpus `336bd906446de4443c68e0696091c581d35ac0ea1b2ece27beb745d7dbaf649f`

**Limitations:** all original parent Q8 receipts share seed 570; there were zero new parent forwards, no real AEROHOUND observations and no native CyberSand execution. The local 98-entry ZIP of trained checkpoints, full synthetic evidence and the independent replay has SHA-256 `cc2ccca5a771432b2400bd948c5645b41380789576e7847571c434a873055879`; it has not been published to immutable GitHub Actions, so the checkpoint is not hosted-qualified and must not be promoted. The upstream D018/D019 source/evidence bundles remain source-pinned separately (SHA-256 `bb43f81e3572ab71599a1512249f8bfc21ab09a53b15d346c43eebd18b38203f` and `12450fec9cbfa5f799ea388d87ca84da91f9d6beb26514f52ac0d272ed893d62`).

Do not turn this into a new PR or workflow; the next meaningful experiment should train branch/coverage-balanced on-policy trajectories in Neurounner only, with a **third untouched family of dynamic-trap worlds** rather than threshold retuning. No CyberSand writes, production changes or paid compute.
