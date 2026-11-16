#!/usr/bin/env python3
"""One-shot, SHA-verified CyberSand research source import. Never edits CyberSand."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = [
 (655,"experiments/driv3r/model_zoo/wide_gru_mz010","MZ-010 wide GRU retained variant"),
 (652,"experiments/driv3r/model_zoo/memory_gate","MZ-008 memory-gate variant"),
 (632,"experiments/driv3r/model_zoo","MZ-001 Pythia pilot historical root files"),
]
PREFIXES=("experiments/driv3r/model_zoo/",)

def git(cwd,*args):
 p=subprocess.run(["git",*args],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode: raise RuntimeError("git "+args[0]+": "+p.stderr.decode(errors="replace")[:1200])
 return p.stdout

def sha(data):
 return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def main():
 if not (ROOT/".git").exists(): raise RuntimeError("Not a Git worktree")
 if (ROOT/"experiments/driv3r/model_zoo/wide_gru_mz010").exists(): raise RuntimeError("Orphan experiment already copied")
 with tempfile.TemporaryDirectory(prefix="driv3r-move-") as tmp:
  source=Path(tmp)/"source"
  git(None,"init","-q",str(source))
  git(source,"remote","add","origin","https://github.com/techrote/cybersand.git")
  refs=["+refs/pull/%d/head:refs/remotes/origin/pr-%d"%(n,n) for n,_,_ in SOURCE]
  try:
   git(source,"fetch","-q","--no-tags","--depth=1","--filter=blob:limit=1048576","origin",*refs)
  except RuntimeError:
   git(source,"fetch","-q","--no-tags","--depth=1","origin",*refs)
  records=[]
  occupied=set()
  for n,path,program in SOURCE:
   head=git(source,"rev-parse","refs/remotes/origin/pr-%d"%n).decode().strip()
   tree=git(source,"ls-tree","-rz",head,"--",path)
   expected={}
   for line in tree.split(b"\0"):
    if not line: continue
    info,p=line.split(b"\t",1)
    mode,kind,obj=info.decode().split()
    if kind=="blob": expected[p.decode()]=(mode,obj)
   if not expected: raise RuntimeError("PR %d: expected prefix %s absent"%(n,path))
   archive=git(source,"archive","--format=tar",head,path)
   copied=[]
   with tarfile.open(fileobj=io.BytesIO(archive),mode="r:") as tar:
    for entry in tar:
     if entry.isdir(): continue
     if not entry.isfile(): raise RuntimeError("Unexpected archive member "+entry.name)
     p=entry.name.rstrip("/")
     if not p.startswith(PREFIXES) or p not in expected or p in occupied:
      raise RuntimeError("Unsafe, unexpected or duplicate file "+p)
     data=tar.extractfile(entry).read()
     digest=sha(data)
     if digest!=expected[p][1]: raise RuntimeError("Blob mismatch "+p)
     dest=ROOT/p
     if dest.exists(): raise RuntimeError("Destination exists "+p)
     dest.parent.mkdir(parents=True,exist_ok=True)
     dest.write_bytes(data)
     if expected[p][0]=="100755": dest.chmod(0o755)
     occupied.add(p)
     copied.append({"path":p,"sha1":digest,"bytes":len(data)})
   if len(copied)!=len(expected): raise RuntimeError("File-count mismatch PR %d"%n)
   records.append({"cybersand_pr":n,"head":head,"source_path":path,"programme":program,"files":copied})
   print("PR #%d: %d blob-verified files, head %s"%(n,len(copied),head),flush=True)
 manifest={"source":"techrote/cybersand","target":"techrote/neurounner",
           "policy":"Research source only; no game code, mergers, CI workflows or external checkpoints",
           "evidence":"Original GitHub Actions artifacts and PR comments stay in source repository",
           "lanes":records}
 dst=ROOT/"docs/driv3r-migration/HISTORICAL_MANIFEST.json"
 dst.parent.mkdir(parents=True,exist_ok=True)
 dst.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print("SUCCESS: %d files across %d programmes"%(len(occupied),len(records)),flush=True)

if __name__=="__main__": main()
