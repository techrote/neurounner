#!/usr/bin/env python3
"""Read-only admission of original W0 ZIPs as EVALUATION REFERENCE, not TRAIN.

Original bytes remain untouched. No model actions are promoted to teacher labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from corpus import ACTIONS, SOURCE, VERSION, canonical


def ingest(archive: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    original=archive.read_bytes(); original_sha=hashlib.sha256(original).hexdigest()
    output=[]; n_model=0; n_control=0; n_events=0; png_count=0
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
        members={f.filename for f in z.infolist() if not f.is_dir()}
        for name in sorted(members):
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe source ZIP member')
            if '/original-png-and-json/' not in name or not name.endswith('.json'):
                continue
            raw=z.read(name);r=json.loads(raw)
            if r.get('contract')!=SOURCE['common_w0_contract']:
                raise ValueError(f'wrong W0 contract: {name}')
            events=r['events']; trajectory=r['trajectory']
            if len(events)!=r['steps_executed'] or len(trajectory)!=len(events)+1:
                raise ValueError(f'invalid W0 step inventory: {name}')
            for tick,event in enumerate(events):
                obs=event['observation']; legal=obs['legal']
                if obs['tick']!=tick or len(obs['local_5x5'])!=5:
                    raise ValueError(f'invalid observation {name}:{tick}')
                if not set(legal).issubset(ACTIONS) or event['applied'] not in legal:
                    raise ValueError(f'host-illegal receipt {name}:{tick}')
                if event['raw_legal'] != (event['raw_choice'] in legal):
                    raise ValueError(f'raw-legality mismatch {name}:{tick}')
                if event['position_after'] != trajectory[tick+1]:
                    raise ValueError(f'position after mismatch {name}:{tick}')
            png_name=name[:-5]+'.png'
            png=z.read(png_name) if png_name in members else None
            if png is not None and not png.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError(f'invalid original PNG magic: {png_name}')
            is_model=any(s in r['mode'].lower() for s in ('sprocket','lfm','model'))
            n_model+=bool(is_model);n_control+=not is_model
            n_events+=len(events);png_count+=png is not None
            output.append({'schema':VERSION,'evidence_class':'read-only-W0-inference-reference-not-training',
                'source_archive_sha256':original_sha,'original_json_member':name,
                'original_json_sha256':hashlib.sha256(raw).hexdigest(),
                'original_png_member':png_name if png else None,
                'original_png_sha256':hashlib.sha256(png).hexdigest() if png else None,
                'origin_engine':r.get('engine'),'origin_mode':r['mode'],
                'origin_contract':r['contract'],'origin_seed':r['seed'],'family':r['family'],
                'source_kind':'model-run' if is_model else 'nonmodel-control',
                'not_for_training':True,'steps':len(events),
                'applied_actions':[event['applied'] for event in events],
                'raw_choices':[event['raw_choice'] for event in events],
                'invalid_raw_count':sum(not event['raw_legal'] for event in events),
                'metrics':r.get('metrics',{})})
    if not output:
        raise ValueError('no original W0 JSON traces found')
    content=''.join(canonical(r)+'\n' for r in output).encode('utf-8')
    result={'archive_sha256':original_sha,'source_name':archive.name,
            'source_ref':SOURCE,'kind':'EVALUATION-REFERENCE-ONLY',
            'records':len(output),'model_runs':n_model,'control_runs':n_control,
            'events':n_events,'original_pngs':png_count,
            'reference_jsonl_sha256':hashlib.sha256(content).hexdigest(),
            'split_admission':'NONE: NEVER appended to TRAIN/DEV/heldout',
            'native_Godot_events':0}
    (destination/'w0-reference.jsonl').write_bytes(content)
    (destination/'w0-reference-manifest.json').write_text(canonical(result)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(canonical(ingest(args.archive,args.output)))


if __name__=='__main__':main()
