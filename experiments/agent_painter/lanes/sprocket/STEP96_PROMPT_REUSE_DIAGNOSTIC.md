# SPROCKET step96 prompt differentiation diagnosis — historical DEVELOPMENT only

**Status:** Source-only, immutable read-only audit. This is not an optimizer run, a model inference, a native painting, a split admission, or a causal experiment. PR #576 stays Draft/HOLD.

Evidence sources: [step16 run 43523253619](https://github.com/techrote/cybersand/actions/runs/43523253619) artifact 14083354475 SHA256 686f67190766d71667b8e78c9a58bdcd121070eff1b75bad21327ac63f3d6b2a; [step96 run 43536717867](https://github.com/techrote/cybersand/actions/runs/43536717867) artifact 14089867633 SHA256 991d2f2c0a578d935c839d5e5b4650ad8b9b9116348a54e873340f59f115ff23. Original native source 633920373a3d406a2ede61ce4137c324f76be23b, manifest SHA256 2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc.

## Observed six-request first-pass results

| Arm | Terminal EOS5 | Strict JSON | Valid program | Compiled request | Actual native model painting |
| --- | ---: | ---: | ---: | ---: | ---: |
| step16 | 2/6 | 1/6 | 0/6 | 0/6 | 0 |
| step96 | 5/6 | 5/6 | 1/6 | 1/6 (2/12 seed plans) | 0 |

**Architecture/Greeble cross-family exact reuse:** Three different original request groups (Architecture bridge/towers, Architecture stacked rooms/shaft, and Greebles open-center) generated precisely the same ordered two box coordinates: [8,24,40,60], [46,24,40,60]. The Greeble output contains room-kind components rather than panels. Bridge/towers changes one component kind to stairs but retains room-only openings/thickness, and omits required stair steps/rise. Stacked rooms/shaft compiles but the geometry is adjacent rooms, not the requested vertical stack and shaft. Exact same coordinates across families indicate poor prompt differentiation; they do NOT prove why it happened.

**Pipework exact reuse:** Both distinct Pipework prompts generated the identical 4-node/3-edge network at coordinates (12,20),(52,20),(52,75),(84,75), differing in diameter 5 versus 3. Both contain the diagonal edge [1,3] from (52,20) to (84,75), violating orthogonal topology, and both declare cycles=0 despite one requesting a closed loop. Both are compiler-refused. The remaining Greeble spine generation reached 512 tokens without EOS5, with repeated panel coordinates.

The separate original native teacher-target budget audit showed all six DEVELOPMENT canonical teacher completions fit within 137–260 SPROCKET tokens including EOS5. That rules out these six canonical targets necessarily exceeding a 512 completion cap; it does not rule out defective generation looping to the cap.

**Conclusion:** Source-observed structured copying and invalid family-specific geometry are more specific failure modes than JSON syntax alone. The audited 16→96 change improved raw syntax but did not produce conforming art. A useful next controlled comparison requires additional shepherd-selected, globally independent, positively native-admitted TRAIN structures, a frozen common DEVELOPMENT suite and strict original raw first-pass evaluation. Do not extend unchanged-corpus optimization or mask failures with repair. No FINAL target/image was opened or used for selection.
