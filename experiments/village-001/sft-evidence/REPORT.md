# VILLAGE-001C — 36-member SFT training report

All 36 canonical VILLAGE-001 members completed the same qualified SmolLM2-135M LoRA recipe on independent GitHub-hosted CPU lanes.

No hidden member/persona description was supplied during held-out evaluation. Member differences arise from the member corpus, deterministic seed, and learned adapter.

## Aggregate

- members: 36/36 terminal successful lanes
- base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- optimizer steps/member: 320
- median training seconds: 306.51
- median sequence tok/s: 128.46
- median peak RSS KiB: 1705050
- resumed members: 0

## Members

| Member | CPU | Seconds | Seq tok/s | Loss first→last | RSS KiB | Adapter archive |
|---|---|---:|---:|---:|---:|---|
| V-G01-A | AMD EPYC 9V45 96-Core Processor | 198.74 | 168.97 | 3.6820→2.9078 | 1610628 | `21aa5d6cb098a47b…` |
| V-G01-B | AMD EPYC 7763 64-Core Processor | 324.60 | 107.03 | 4.5087→2.0673 | 1606000 | `3e67b791fa846306…` |
| V-G01-C | Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | 269.13 | 135.36 | 4.0098→3.0264 | 1761680 | `4c3e49cd3ebd5699…` |
| V-G02-A | AMD EPYC 9V45 96-Core Processor | 235.87 | 177.75 | 3.9899→3.0026 | 1746100 | `e7b2537434db13c7…` |
| V-G02-B | AMD EPYC 7763 64-Core Processor | 374.74 | 106.06 | 3.9754→2.2613 | 1727460 | `6f28ce18c1fd1359…` |
| V-G02-C | AMD EPYC 9V74 80-Core Processor | 308.46 | 134.81 | 3.6555→2.1456 | 1743112 | `20163746f2d43876…` |
| V-G03-A | AMD EPYC 7763 64-Core Processor | 352.91 | 110.53 | 4.3324→2.3516 | 1741828 | `89274f05bdc8af13…` |
| V-G03-B | AMD EPYC 9V74 80-Core Processor | 299.20 | 133.37 | 4.2039→2.2226 | 1704916 | `06ca45c71dfec8bc…` |
| V-G03-C | AMD EPYC 9V45 96-Core Processor | 218.08 | 182.90 | 4.6161→2.0891 | 1733792 | `5269f0827166a284…` |
| V-G04-A | AMD EPYC 7763 64-Core Processor | 360.93 | 114.13 | 3.4093→1.4157 | 1752304 | `7f8d34d66373cb93…` |
| V-G04-B | AMD EPYC 7763 64-Core Processor | 344.63 | 113.16 | 2.8314→2.0610 | 1705184 | `9cb535e05145c648…` |
| V-G04-C | AMD EPYC 9V45 96-Core Processor | 224.33 | 176.64 | 3.2169→1.9238 | 1736280 | `5e79fb2c6e0e3373…` |
| V-G05-A | AMD EPYC 7763 64-Core Processor | 341.84 | 112.94 | 4.2546→2.8691 | 1742312 | `64aae22f25c547ec…` |
| V-G05-B | AMD EPYC 9V74 80-Core Processor | 293.67 | 134.28 | 3.4682→2.4803 | 1668428 | `4eafb68d72438717…` |
| V-G05-C | AMD EPYC 7763 64-Core Processor | 347.04 | 112.11 | 3.6155→2.2649 | 1682208 | `aa32fae4ba324f74…` |
| V-G06-A | Intel(R) Xeon(R) 6973P-C | 184.46 | 231.67 | 3.8339→2.0296 | 1749412 | `eea3fe0adba275de…` |
| V-G06-B | AMD EPYC 9V74 80-Core Processor | 308.49 | 134.78 | 3.5301→2.3814 | 1740080 | `cde7d3162e648ef4…` |
| V-G06-C | AMD EPYC 7763 64-Core Processor | 394.11 | 112.96 | 4.4428→2.3476 | 1874356 | `22744a7b32b5e2d7…` |
| V-G07-A | AMD EPYC 7763 64-Core Processor | 327.69 | 111.61 | 3.9864→2.3706 | 1654976 | `b25f69a0586d11ee…` |
| V-G07-B | AMD EPYC 7763 64-Core Processor | 324.04 | 109.84 | 3.7004→2.4935 | 1633616 | `b717c11fec46ba2d…` |
| V-G07-C | Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | 259.89 | 139.28 | 3.1579→2.4872 | 1662848 | `e42338b738591181…` |
| V-G08-A | AMD EPYC 9V45 96-Core Processor | 203.49 | 177.15 | 4.5768→1.9915 | 1650504 | `00f4adae21322c41…` |
| V-G08-B | AMD EPYC 7763 64-Core Processor | 326.45 | 110.93 | 3.4966→1.8880 | 1663984 | `6dcbfecb60faccd2…` |
| V-G08-C | AMD EPYC 7763 64-Core Processor | 324.92 | 112.12 | 4.0893→2.1636 | 1647504 | `cba725aa19eb9d21…` |
| V-G09-A | AMD EPYC 9V74 80-Core Processor | 272.06 | 133.04 | 4.1609→2.1133 | 1701168 | `58d9c18867f902b1…` |
| V-G09-B | AMD EPYC 9V74 80-Core Processor | 285.08 | 128.10 | 4.0181→1.9736 | 1674756 | `e6a86b27a103a527…` |
| V-G09-C | AMD EPYC 9V74 80-Core Processor | 275.99 | 130.25 | 4.1022→2.0570 | 1655956 | `8e36f62c47a156ee…` |
| V-G10-A | AMD EPYC 9V74 80-Core Processor | 310.25 | 128.83 | 4.5925→1.2353 | 1715392 | `4c4215c50b949746…` |
| V-G10-B | AMD EPYC 7763 64-Core Processor | 356.62 | 113.27 | 4.1797→2.8927 | 1719432 | `709fe9f86407860a…` |
| V-G10-C | AMD EPYC 9V45 96-Core Processor | 227.64 | 182.14 | 4.3530→2.8701 | 1712312 | `d48f777f70cd1e04…` |
| V-G11-A | AMD EPYC 7763 64-Core Processor | 304.56 | 109.36 | 3.1628→1.3405 | 1588560 | `ac920bd0dd18aeec…` |
| V-G11-B | AMD EPYC 9V45 96-Core Processor | 188.79 | 174.30 | 5.1513→1.9679 | 1610732 | `3d9fe49247566687…` |
| V-G11-C | AMD EPYC 7763 64-Core Processor | 303.28 | 110.43 | 3.8924→2.1926 | 1608984 | `fc7314ed678edd06…` |
| V-G12-A | AMD EPYC 7763 64-Core Processor | 355.74 | 114.12 | 3.4990→2.8301 | 1701464 | `ae28bb7316c0e933…` |
| V-G12-B | INTEL(R) XEON(R) PLATINUM 8573C | 183.39 | 218.81 | 3.9105→2.4454 | 1712464 | `d6830599c344a189…` |
| V-G12-C | AMD EPYC 7763 64-Core Processor | 448.85 | 93.21 | 3.6698→2.2114 | 1803968 | `a6c5f2edb7449270…` |

Raw held-out responses and full loss traces are retained in `results/*.json`. Adapter bytes are published as release assets; their archive and unpacked-directory hashes are recorded in `adapter-release.json`.
