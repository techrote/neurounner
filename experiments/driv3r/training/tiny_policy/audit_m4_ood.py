"""Offline whole-archive M4 audit; verifies hashes, decisions, original pixels and summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import ACTIONS, FAMILIES, raster
from .m4_ood import SEEDS, TOPOLOGIES, WINDOW, manifest, replay


def audit(root:Path):
    recorded=json.loads((root/'manifest.json').read_text())
    real=manifest(root)
    if recorded!=real:raise ValueError('Archive SHA-256 manifest mismatch')
    report=json.loads((root/'report.json').read_text())
    rows=json.loads((root/'rollout_rows.json').read_text())
    if report['status']!='COMPLETED' or report['optimizer_updates']!=0:
        raise ValueError('Unqualified run or unexpected optimizer')
    if len(rows)!=len(SEEDS)*len(FAMILIES)*8:raise ValueError('Missing/duplicate rows')
    actual=set(); nsteps=0
    for row in rows:
        seed,family,policy=row['seed'],row['family'],row['policy']
        key=(seed,family,policy)
        if key in actual:raise ValueError('Duplicated trajectory identity')
        actual.add(key)
        path=root/'rollouts'/f'{seed}-{family}-{policy}'
        receipt=json.loads(path.with_suffix('.json').read_text())
        if (receipt['seed'],receipt['family'],receipt['policy'])!=key:
            raise ValueError('Receipt identity mismatch')
        w=replay(receipt)
        nsteps+=len(receipt['events'])
        if receipt['steps']>WINDOW or receipt['steps']<1:raise ValueError('Invalid budget')
        for event in receipt['events']:
            if event['action'] not in event['legal'] or event['action'] not in ACTIONS:
                raise ValueError('Illegal applied action')
        with Image.open(path.with_suffix('.png')) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')),np.asarray(raster(w,3))):
                raise ValueError('Original PNG mismatch')
        for k in ('steps','cells','branches','retraces','dead_ends','stopped','reachable'):
            if row[k]!=receipt[k]:raise ValueError('Metrics mismatch')
    if len(actual)!=288:raise ValueError('Incomplete policy/world product')
    return {'status':'PASS','rollouts':len(actual),'verified_actions':nsteps,
            'verified_files':len(real),'original_pngs':len(actual),
            'unique_seeds':len(set(x[0] for x in actual)),
            'topologies':list(TOPOLOGIES)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True,type=Path)
    result=audit(parser.parse_args().archive)
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
