#!/usr/bin/env python3
"""DG020 H44 exact last-layer infeasibility certificate: Python stdlib only.

Signed float32 hidden-route vectors reconstructed from genuinely trained H40
donor/receiver checkpoints and exposed observations. A positive linear
dependence of signed vectors proves no eight-weight last-layer setting can
assign strictly positive margins to ALL of these examples while hidden
activations remain frozen. This says nothing about unrestricted 209-parameter
training or general physical-law inheritance. The complete neural replay and
bitwise feature-origin audit remain in the owner H44 evidence capsule.
"""
from fractions import Fraction
import struct

SUPPORT = {
 '26011639': """
3f264bee 3ea57770 3ea59c40 b9951280 80000000 bad97800 80000000 ba6a9780
3f2512bc bef813c8 be24d534 bb148a88 00000000 3c789010 00000000 38ed4900
3fa7866a bea5f960 bea6c2b0 bb3c8900 b9b97d68 3cf3b9c4 00000000 398ce180
3fa6ad78 bfa27fc0 bfa245ec bf7309d6 be9937dc 3ce29148 00000000 baeee9b0
bfa735b0 3fa667a0 3fa62670 3f7bc000 3eaa9ee8 bc6c6ad4 00000000 3afe77e0
bf2661a8 3ef81a68 3e24f6c0 3b15a330 00000000 bc7a5160 00000000 b8f67800
3f250e14 bef82e60 be255cc0 bb174688 00000000 3c7ae590 00000000 3913ef00
bf252ef8 3ef82b20 3e2549c0 3b164210 00000000 bc7a0710 00000000 b9091880
""",
 '26011641': """
3f25881c befa2884 be247018 80000000 3a51ceb0 80000000 80000000 80000000
bfa5d738 3fa73040 3fa5b448 3f79d730 3eaaa500 80000000 80000000 80000000
3fa5a9b2 3efd3ba0 3e237fb0 00000000 3b25315c 00000000 00000000 00000000
bf272bd8 3efa0f9c 3e24358c 80000000 ba5a1390 80000000 80000000 80000000
3f25d00c bef99eb0 be232ff0 00000000 3a4ef3b0 00000000 00000000 00000000
3fa57406 bfa2fb68 bfa19244 bf719788 be99c694 00000000 00000000 00000000
"""
}

def positive_exact_null(transposed):
    a = [list(row) for row in transposed]
    rows, cols = len(a), len(a[0])
    pivots = []
    r = 0
    for c in range(cols):
        k = next((j for j in range(r, rows) if a[j][c]), None)
        if k is None:
            continue
        a[r], a[k] = a[k], a[r]
        d = a[r][c]
        a[r] = [v / d for v in a[r]]
        for j in range(rows):
            if j != r and a[j][c]:
                mult = a[j][c]
                a[j] = [a[j][i] - mult * a[r][i] for i in range(cols)]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [i for i in range(cols) if i not in pivots]
    assert len(free) == 1, 'Unexpected nullspace dimension'
    v = [Fraction(0) for _ in range(cols)]
    v[free[0]] = Fraction(1)
    for j, pivot in enumerate(pivots):
        v[pivot] = -a[j][free[0]]
    assert all(coeff > 0 for coeff in v), 'Not a positive dependence'
    assert all(sum(row[j] * v[j] for j in range(cols)) == 0 for row in transposed)
    return v

def main():
    for seed, raw in SUPPORT.items():
        vectors = [[Fraction.from_float(struct.unpack('!f', bytes.fromhex(token))[0])
                    for token in line.split()]
                   for line in raw.strip().splitlines()]
        assert all(len(row) == 8 for row in vectors)
        v = positive_exact_null(list(zip(*vectors)))
        assert all(sum(row[i] * v[j] for j, row in enumerate(vectors)) == 0 for i in range(8))
        print(f'{seed}: EXACT POSITIVE NULL CERTIFICATE PASS; {len(v)} supporting signed float32 vectors')
    print('H44 FROZEN-HIDDEN OUTPUT-HEAD STRICT-SEPARATION INFEASIBLE')

if __name__ == '__main__':
    main()
