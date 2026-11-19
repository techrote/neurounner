#!/usr/bin/env python3
"""DG020 H45: verify bit-exact numerical receiver patches against frozen originals.

Original frozen H40 genuine receiver files and H45 candidate bytes are supplied
in the owner H45 reproducibility capsule. This checks numeric tensor identity;
rewriting a torch.save file is not promised to reproduce its original byte SHA.
"""
import sys, json, hashlib, struct
from pathlib import Path
import torch

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(root):
    rec=json.loads((root/'H45_RECEIVER_FLOAT32_BIT_PATCHES.json').read_text())
    loc={'26011639':'h36-output/receiver-h36-anchored-26011639.pt',
         '26011640':'h36-output/receiver-h36-anchored-26011640.pt',
         '26011641':'h39-output/receiver-h39-margin-sweep-best-26011641.pt'}
    candidates={
      '26011639':'h45-output/h45-possible-feasible-209param-receiver-26011639-knee2.pt',
      '26011640':'h45-output/h45-possible-feasible-209param-receiver-26011640-knee2.pt',
      '26011641':'h45-output/h45-possible-feasible-209param-receiver-26011641-two-knee-2-0.pt'}
    for seed,entry in rec.items():
        original=root/loc[seed]
        selected=root/candidates[seed]
        assert sha(original)==entry['source_checkpoint_sha256']
        assert sha(selected)==entry['selected_h45_checkpoint_sha256']
        state=torch.load(original,weights_only=True,map_location='cpu')
        expected=torch.load(selected,weights_only=True,map_location='cpu')
        assert set(entry['bitwise_patch']) <= {'tile.0.bias','tile.4.weight'}
        for name,changes in entry['bitwise_patch'].items():
            for e in changes:
                i=int(e['flat_index'])
                b=bytes.fromhex(e['bits_le_hex'])
                assert len(b)==4
                value=struct.unpack('<f',b)[0]
                assert struct.pack('<f',value)==b
                state[name].reshape(-1)[i]=value
        assert all(torch.equal(state[k],expected[k]) for k in expected),seed
        assert sum(x.numel() for x in state.values())==209
        print(seed,'BITEXACT_NUMERICAL_TENSOR_PATCH_PASS')

if __name__=='__main__':
    main(Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve())
