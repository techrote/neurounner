# VILLAGE-001C — 36-member SFT training report

All 36 canonical VILLAGE-001 members completed the same qualified SmolLM2-135M LoRA recipe on independent GitHub-hosted CPU lanes.

No hidden member/persona description was supplied during held-out evaluation. Member differences arise from the member corpus, deterministic seed, and learned adapter.

## Aggregate

- members: 36/36 terminal successful lanes
- base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- optimizer steps/member: 320
- median training seconds: 313.05
- median sequence tok/s: 114.67
- median peak RSS KiB: 1700878
- resumed members: 0

## Members

| Member | CPU | Seconds | Seq tok/s | Loss first→last | RSS KiB | Adapter archive |
|---|---|---:|---:|---:|---:|---|
| V-G01-A | AMD EPYC 9V74 80-Core Processor | 262.46 | 127.94 | 3.6820→2.9078 | 1601440 | `8e9e37d434258fe0…` |
| V-G01-B | INTEL(R) XEON(R) PLATINUM 8573C | 185.58 | 187.21 | 4.5087→2.0673 | 1595420 | `658c456d36e23f11…` |
| V-G01-C | AMD EPYC 7763 64-Core Processor | 326.43 | 111.60 | 4.0098→3.0266 | 1753616 | `6484916efd90331d…` |
| V-G02-A | AMD EPYC 7763 64-Core Processor | 372.18 | 112.65 | 3.9899→3.0026 | 1726632 | `b89ee449c9589457…` |
| V-G02-B | AMD EPYC 7763 64-Core Processor | 357.64 | 111.13 | 3.9754→2.2613 | 1751616 | `c5cfbafc70c775e3…` |
| V-G02-C | AMD EPYC 7763 64-Core Processor | 368.04 | 112.98 | 3.6555→2.1455 | 1743444 | `257322d41e21eff3…` |
| V-G03-A | AMD EPYC 7763 64-Core Processor | 344.29 | 113.29 | 4.3324→2.3516 | 1723948 | `fae1fda94c69c93d…` |
| V-G03-B | INTEL(R) XEON(R) PLATINUM 8573C | 206.87 | 192.89 | 4.2039→2.2226 | 1722612 | `9bb6bb160ea7e7f6…` |
| V-G03-C | AMD EPYC 9V45 96-Core Processor | 223.35 | 178.59 | 4.6161→2.0891 | 1736760 | `f8c79e2da982876e…` |
| V-G04-A | AMD EPYC 7763 64-Core Processor | 363.21 | 113.41 | 3.4093→1.4157 | 1752868 | `bd80f2e0d3ddf7e2…` |
| V-G04-B | AMD EPYC 9V45 96-Core Processor | 201.66 | 193.39 | 2.8314→2.0610 | 1741036 | `1b7515e962588ec0…` |
| V-G04-C | AMD EPYC 7763 64-Core Processor | 348.87 | 113.58 | 3.2169→1.9238 | 1735096 | `c27cb892c69aaa20…` |
| V-G05-A | Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | 269.48 | 143.27 | 4.2546→2.8691 | 1762036 | `86788f6fdfd757b8…` |
| V-G05-B | INTEL(R) XEON(R) PLATINUM 8573C | 210.83 | 187.04 | 3.4682→2.4803 | 1686508 | `90bbdf83f5737b5c…` |
| V-G05-C | AMD EPYC 9V74 80-Core Processor | 289.31 | 134.49 | 3.6155→2.2649 | 1668624 | `c818bb0c77aa6c93…` |
| V-G06-A | AMD EPYC 7763 64-Core Processor | 378.51 | 112.90 | 3.8339→2.0296 | 1752232 | `66b30b56477a623a…` |
| V-G06-B | AMD EPYC 9V45 96-Core Processor | 224.55 | 185.17 | 3.5301→2.3814 | 1752856 | `82368475bb9b1113…` |
| V-G06-C | AMD EPYC 7763 64-Core Processor | 386.52 | 115.18 | 4.4428→2.3476 | 1881740 | `43d52ddaf7a539ef…` |
| V-G07-A | AMD EPYC 7763 64-Core Processor | 323.80 | 112.95 | 3.9864→2.3706 | 1676412 | `20064d30ac22cf9b…` |
| V-G07-B | AMD EPYC 9V74 80-Core Processor | 269.36 | 132.14 | 3.7004→2.4934 | 1623800 | `be3b296a89a4afb3…` |
| V-G07-C | AMD EPYC 7763 64-Core Processor | 321.06 | 112.74 | 3.1579→2.4872 | 1638296 | `c754cd36ec82fbae…` |
| V-G08-A | AMD EPYC 9V45 96-Core Processor | 195.58 | 184.31 | 4.5768→1.9915 | 1620632 | `1f55d08f68bee9dd…` |
| V-G08-B | AMD EPYC 7763 64-Core Processor | 326.43 | 110.93 | 3.4966→1.8880 | 1662688 | `f1eedea1251c5587…` |
| V-G08-C | AMD EPYC 7763 64-Core Processor | 326.66 | 111.52 | 4.0893→2.1636 | 1637652 | `2878e0d66eb07c25…` |
| V-G09-A | AMD EPYC 9V45 96-Core Processor | 199.42 | 181.49 | 4.1609→2.1133 | 1693100 | `0cf14d6cb6cf5e28…` |
| V-G09-B | AMD EPYC 9V45 96-Core Processor | 219.26 | 166.55 | 4.0181→1.9736 | 1668968 | `190284c9598c0589…` |
| V-G09-C | AMD EPYC 7763 64-Core Processor | 322.21 | 111.56 | 4.1022→2.0570 | 1694380 | `2caa935d79396584…` |
| V-G10-A | AMD EPYC 9V45 96-Core Processor | 228.30 | 175.07 | 4.5925→1.2353 | 1722336 | `ec8be5cde8461dea…` |
| V-G10-B | AMD EPYC 7763 64-Core Processor | 361.98 | 111.60 | 4.1797→2.8927 | 1743268 | `567e5cd31e510201…` |
| V-G10-C | AMD EPYC 7763 64-Core Processor | 361.42 | 114.72 | 4.3530→2.8701 | 1691344 | `e64f247e6c3a7cf3…` |
| V-G11-A | AMD EPYC 9V74 80-Core Processor | 258.20 | 129.00 | 3.1628→1.3405 | 1557708 | `1f1cf4867d2e4ed0…` |
| V-G11-B | AMD EPYC 7763 64-Core Processor | 305.03 | 107.88 | 5.1513→1.9679 | 1600916 | `132733336bdf88a2…` |
| V-G11-C | AMD EPYC 7763 64-Core Processor | 304.52 | 109.98 | 3.8924→2.1926 | 1619840 | `a804978b17b2afc8…` |
| V-G12-A | AMD EPYC 9V45 96-Core Processor | 235.45 | 172.43 | 3.4990→2.8300 | 1707376 | `2b761801dca06984…` |
| V-G12-B | AMD EPYC 7763 64-Core Processor | 350.13 | 114.61 | 3.9105→2.4454 | 1684172 | `c0b67d37fb96e49a…` |
| V-G12-C | AMD EPYC 7763 64-Core Processor | 367.22 | 113.93 | 3.6698→2.2114 | 1813664 | `de3a6ba1eec5c828…` |

Raw held-out responses and full loss traces are retained in `results/*.json`. Adapter bytes are published as release assets; their archive and unpacked-directory hashes are recorded in `adapter-release.json`.
