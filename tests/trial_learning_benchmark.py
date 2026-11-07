"""Offline adversarial graph fixture; held-out labels never construct mutations."""
import sys,tempfile,json,contextlib,io,time,tracemalloc,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import CrystalLattice,EDGE_IS_A,CrystalReason
from membrane import LanguageMembrane
from imitation_learning import ExperimentalLearningLoop

def run(seed):
 random.seed(seed);start=time.perf_counter();tracemalloc.start()
 with tempfile.TemporaryDirectory() as d:
  g=CrystalLattice();m=LanguageMembrane(g,data_dir=d)
  subjects=('мирак','новал','третак');objects=('луны','споры','камни')
  for s in subjects:g.connect(s,'классА',70,EDGE_IS_A,CrystalReason('input',source_label='fixture classes',metadata={'status':'observed'}));g.connect(s,'ошибка',100)
  for o in objects:g.connect(o,'классБ',70,EDGE_IS_A,CrystalReason('input',source_label='fixture classes',metadata={'status':'observed'}))
  # Weak SYN supplies candidate paths; no predicate facts are provided.
  for s,o in zip(subjects[1:],objects[1:]):g.connect(s,o,10)
  loop=ExperimentalLearningLoop(m)
  baseline=[loop._generate('что поглощает '+s+'?',loop.store['model'])[0]['object'] for s in subjects]
  first=loop.attempt('что поглощает мирак?');negative=loop.feedback('incorrect')
  correction=loop.feedback('reference_answer','мирак поглощает луны')
  retry=loop.attempt('что поглощает мирак?')
  heldout=[]
  for s,o in zip(subjects[1:],objects[1:]):
   attempt=loop.attempt('что поглощает '+s+'?');correct=attempt['selected']['object']==o
   heldout.append(dict(subject=s,expected=o,attempt=attempt,correct=correct))
   loop.feedback('correct' if correct else 'incorrect')
  peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
  return dict(seed=seed,first=first,negative=negative,correction=correction,retry=retry,heldout=heldout,
    baseline_accuracy=sum(a==b for a,b in zip(baseline,objects))/3,heldout_accuracy=sum(x['correct'] for x in heldout)/2,
    status=loop.status(),seconds=time.perf_counter()-start,peak_bytes=peak,
    verified_false=sum(m.respond('что поглощает '+s+'?','verified')['mode']=='Verified' for s in subjects))
with contextlib.redirect_stdout(io.StringIO()):rows=[run(s) for s in (17,29,43)]
print(json.dumps(rows,ensure_ascii=False,indent=2))
