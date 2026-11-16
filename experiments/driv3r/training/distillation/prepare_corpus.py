"""Read two immutable real d1 Frankenbrain Actions ZIPs into a source-pinned text-action corpus."""
import hashlib,json,zipfile,sys
from pathlib import Path
SHA={'greyhound':'e4bf6b63997967cb0ccc1b4a34e517c7abb62a79d5dc8069e93b9a02371ecff3',
'branchwitch':'0ca2cf83eca44e3462a29ba0a27ae41e62d342bb8a4224f677539a26b68bc181'}
ARTIFACTS={'greyhound':14181164165,'branchwitch':14182441899}
ACTIONS=('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def extract(archives):
 out=[];seen=set()
 for name in ('greyhound','branchwitch'):
  path=Path(archives[name]);raw=path.read_bytes()
  if sha(raw)!=SHA[name]:raise ValueError('Original archive SHA mismatch: '+name)
  with zipfile.ZipFile(path) as z:
   if z.testzip():raise ValueError('Corrupt original archive')
   comparison=json.loads(z.read('inference/comparison.json'))
   if comparison['mode']!=name or not comparison['actual_systemone_inference'] or comparison['training'] or comparison['native_workbench']:raise ValueError('Unexpected producer')
   for kind in (('stock','mutant') if name=='greyhound' else ('mutant',)):
    teacher='stock' if kind=='stock' else name
    for family in ('greebles','pipework','architecture'):
     src=f'inference/{teacher}/{teacher}-{family}-choices.json'
     records=json.loads(z.read(src))
     if len(records)!=16:raise ValueError('Expected original 16-turn episode')
     for i,r in enumerate(records):
      obs=r['observation'];probabilities=r['parsed']['probabilities']
      if r['status']!='ok' or r['choice'] not in obs['legal'] or r['choice'] not in ACTIONS or i!=obs['tick']:raise ValueError('Malformed original model decision')
      if sorted(probabilities)!=sorted(obs['legal']) or abs(sum(probabilities.values())-1)>1e-5:raise ValueError('Bad original probabilities')
      key=(teacher,family,i)
      if key in seen:raise ValueError('Duplicated original record')
      seen.add(key)
      out.append({'schema':'cybersand.driv3r.distill.original-teacher/1','source_kind':'original_model_inference',
       'teacher':teacher,'family':family,'source_seed':570,'tick':i,'observation':obs,'action':r['choice'],
       'probabilities':{a:probabilities.get(a,0) for a in ACTIONS},
       'original_record_sha256':sha(canonical(r)), 'source_archive_sha256':SHA[name],
       'source_artifact_id':ARTIFACTS[name],'source_path':src,
       'producer_source_sha':comparison['source_sha'],'model_sha256':comparison['q8_sha256'],
       'producer_llama_sha':comparison['llama_source_commit']})
 if len(out)!=144:raise ValueError('Expected 144 original calls')
 return out
def main():
 if len(sys.argv)!=4:raise SystemExit('Usage: prepare_corpus.py GREYHOUND.zip BRANCHWITCH.zip OUTPUT.jsonl')
 rows=extract({'greyhound':Path(sys.argv[1]),'branchwitch':Path(sys.argv[2])})
 dst=Path(sys.argv[3]);dst.parent.mkdir(parents=True,exist_ok=True)
 dst.write_bytes(b''.join(canonical(x)+b'\n' for x in rows))
 print(json.dumps({'original_model_decisions':len(rows),'source_dataset_sha256':sha(dst.read_bytes()),'source_artifacts':ARTIFACTS}))
if __name__=='__main__':main()
