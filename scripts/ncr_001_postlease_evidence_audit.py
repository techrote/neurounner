#!/usr/bin/env python3
"""NCR-001 post-lease original Actions evidence: byte readback, negatives; NO training."""
import base64, copy, gzip, hashlib, io, json, tarfile
from pathlib import Path
E=Path(__file__).resolve().parents[1]/'experiments/ncr-001/controls-evidence/postlease-sprocket-20261028'
HEAD='2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80'
SHA='edc47d6a3416a57bc7fc4fff2fb8ddc129e2370574258c04c3d57bd4f6c57b24'
WEIGHTS={'chat':('sprocket_chat','chat/model.safetensors','551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97'),'instruct':('sprocket_instruct','model.safetensors','1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864')}
ZIPS={'source':'11c9e4a5f009b074243bc462b32e4e830d548b2463891583f3aed4270096228e','chat':'870ab4aa2df927827659c816d8f2d9d31974e5cb8f57a5925aa72a4dd1ecbe7f','instruct':'f8602daca392f89739637c4c5dbb04eb392c38c19f7ac9bb665b8507eca27daf'}
def digest(b):return hashlib.sha256(b).hexdigest()
def ok(v,msg):
 if not v:raise AssertionError(msg)
def load():
 m=json.loads((E/'manifest.json').read_text());z=base64.b64decode((E/'original-receipts.tar.gz.b64').read_text().strip(),validate=True)
 ok(digest(z)==SHA==m['receipt_bundle_sha256'],'archive SHA');raw=gzip.decompress(z)
 ok(digest(raw)==m['receipt_bundle_uncompressed_tar_sha256'],'tar SHA');files={}
 with tarfile.open(fileobj=io.BytesIO(raw)) as t:
  for v in t:
   ok(v.isfile() and '..' not in v.name and v.name not in files,'unsafe archive')
   files[v.name]=t.extractfile(v).read()
 ok(set(files)=={n for n in m['original_files'] if not n.endswith('.pt')},'missing original records')
 for n,b in files.items():
  ok(len(b)==m['original_files'][n]['size'] and digest(b)==m['original_files'][n]['sha256'],'original byte mismatch '+n)
 return m,files
def audit(m,f):
 ok(m['source_head']==HEAD and m['run_id']==40681920472 and m['run_attempt']==1,'source/run mismatch')
 ok(m['training_authorized'] is False and m['independent_acceptance'] is False and m['classification'].endswith('FINAL_72_BLOCKED'),'admission claim')
 for k,v in ZIPS.items():ok(m['artifacts'][k]['zip_sha256']==v,'artifact ZIP mismatch')
 result={}
 for variant,(lane,weight_path,weight_sha) in WEIGHTS.items():
  p,c=[json.loads(f[variant+'/'+phase+'.json']) for phase in ('producer','consumer')]
  ok(p['status']=='PASS_SOURCE_MATCHED_POSTLEASE_PRODUCER' and c['status']=='PASS_SOURCE_MATCHED_POSTLEASE_CONSUMER','original run failure')
  for phase,r in (('producer',p),('consumer',c)):
   ok(r['lane']==lane and r['phase']==phase and r['original_frozen_model_executed'] is True,'wrong physical lane')
   ok(r['optimizer_steps']==0,'optimizer step')
   for flag in ('train_authorized','training_performed','comparative_training_performed','heldout_opened','actual_host_final_admission','required_learning_update_loss_checks_pass','tool_memory_semantics_qualified'):ok(r[flag] is False,'false '+flag)
   ok(r['runner']['head_sha']==HEAD and r['runner']['run_id']=='40681920472' and r['runner']['attempt']=='1','runner mismatch')
   ok(r['runner']['torch']=='2.14.1+cpu' and r['runner']['transformers']=='4.57.6','version drift')
   a=r['source_identity']
   ok(a['weight_path']==weight_path and a['weight_sha256']==weight_sha and a['taps']==[7,16,25] and a['revision']=='eb157d4ddddc4619e661d4a2426bb2dd000fc668','checkpoint substitution')
   ok(r['source_sha256']['experiments/ncr-001/protocol.json']=='a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4','protocol SHA')
  ok(p['runner']['pid']!=c['runner']['pid'] and c['producer_pid']==p['runner']['pid'] and c['runner']['host']==p['runner']['host'],'not cold separate interpreter')
  ok(p['source_sha256']==c['source_sha256'] and p['verified_source_files']==c['verified_source_files'],'cold source mismatch')
  ok(p['verified_source_files'][weight_path]['sha256']==weight_sha,'full checkpoint bytes')
  ok(p['in_memory_frozen_before_sha256']==p['in_memory_frozen_after_sha256'],'frozen weights changed')
  ok(p['context_units']==[512,2048] and p['nca_parameters']==658904 and p['mlp_parameters']==658432,'geometry/context')
  x=p['checks']
  for adapter in ('nca','mlp'):
   for suffix in ('zero_noop_16','zero_noop_128','disabled_16','disabled_128','left_masked','right_masked','suffix_invariance','A_B_A_reset','full_vs_cached_512','full_vs_cached_2048'):
    r=x[adapter+'_'+suffix];ok(r['pass'] is True and r['finite'] is True and r['max_abs']<=(0.0005 if 'cached' in suffix else 0),'causal/cache failure '+suffix)
   ok(x[adapter+'_hf_generate_zero_cache']['exact'] is True,'generation parity')
   ok(x[adapter+'_manual_untrained_effect']['learned'] is False,'false learning')
   ok(x[adapter+'_blocks_'+('mlp' if adapter=='nca' else 'nca')]['foreign_disabled_refused'] is True,'cross-hook contamination')
   ok(digest(f[variant+'/'+adapter+'/metadata.json'])==x[adapter+'_adapter_metadata_sha256'],'adapter metadata SHA')
   ok(m['original_files'][variant+'/'+adapter+'/adapter.pt']['sha256']==x[adapter+'_adapter_state_sha256'],'binary state SHA')
   cold=c['checks'][adapter+'_separate_python_reload']
   ok(cold['byte_exact'] is True and cold['sha256']==cold['expected_sha256']==x[adapter+'_cold_target_sha256'],'cold replay')
  ok(x['raw_control_ids_cache_transport_only']['pass'] is True,'control ID transport')
  result[lane]={'nca_2048_max_abs':x['nca_full_vs_cached_2048']['max_abs'],'mlp_2048_max_abs':x['mlp_full_vs_cached_2048']['max_abs']}
 ok(b'Ran 7 tests' in f['source/source-tests.log'] and b'OK' in f['source/source-tests.log'],'7 source tests')
 ok(b'Ran 10 tests' in f['source/shared-lease-tests.log'] and b'OK' in f['source/shared-lease-tests.log'],'10 lease tests')
 return result
def negative(m,f):
 cases=[('chat','producer',lambda d:d['source_identity'].update(weight_path='model.safetensors')),('instruct','consumer',lambda d:d.update(train_authorized=True)),('chat','consumer',lambda d:d.update(actual_host_final_admission=True)),('instruct','producer',lambda d:d['checks']['nca_full_vs_cached_2048'].update(max_abs=0.001)),('chat','consumer',lambda d:d['checks']['mlp_separate_python_reload'].update(byte_exact=False)),('instruct','producer',lambda d:d.update(tool_memory_semantics_qualified=True)),('chat','producer',lambda d:d.update(optimizer_steps=1))]
 for lane,phase,mutation in cases:
  name=lane+'/'+phase+'.json';g=dict(f);obj=json.loads(g[name]);mutation(obj);g[name]=json.dumps(obj).encode()
  try:audit(m,g)
  except AssertionError:continue
  raise AssertionError('negative control missed '+name)
 h=copy.deepcopy(m);h['artifacts']['chat']['zip_sha256']='0'*64
 try:audit(h,f)
 except AssertionError:return len(cases)+1
 raise AssertionError('negative control missed ZIP corruption')
if __name__=='__main__':
 m,f=load();result=audit(m,f);print(json.dumps({'result':'PHYSICAL_PREPASS_PASS__FINAL_72_BLOCKED','lanes':result,'negative_controls_passed':negative(m,f),'train_authorized':False,'optimizer_steps':0,'independent_acceptance':False},sort_keys=True,indent=2))
