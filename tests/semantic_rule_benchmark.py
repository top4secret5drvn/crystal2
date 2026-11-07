"""Fixed audit set; test answers never enter rule candidate construction."""
import sys,json,tempfile,time,tracemalloc,contextlib,io,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import CrystalLattice
from membrane import LanguageMembrane
from semantic_rules import state,interpret,questions,correct
from structured_observations import parse_definition
CASES=[('Рубин — это минерал','рубин','IS_A','минерал'),('Датчик входит в состав панели','датчик','PART_OF','панели'),('Станция состоит из отсеков','отсеков','PART_OF','станция'),('Нагрев вызывает расширение','нагрев','CAUSE','расширение'),('Сбой возникает вследствие перегрева','сбой','EFFECT','перегрева'),('Запуск требует условия готовности','запуск','COND','готовности'),('Список за исключением заголовка','список','EXCEPT','заголовка'),('Модуль аналогичен блоку','модуль','ANALOG','блоку'),('Прибор предназначен для контроля','прибор','GOAL','контроля'),('Вода — холодная',None,None,None),('Нагрев не вызывает дождь',None,None,None),('Если дождь, то мокро',None,None,None),('Узел собран из секций','секций','PART_OF','узел')]

def run(mode,seed):
 random.seed(seed);start=time.perf_counter();tracemalloc.start()
 with tempfile.TemporaryDirectory() as d:
  g=CrystalLattice();m=LanguageMembrane(g,data_dir=d);s=state(g);mutations=0;corrections=0
  import copy
  initial=copy.deepcopy(s['rules'])
  if mode in ('imitation','mutations','active'):
   for text in ('Двигатель состоит из деталей','Прибор состоит из модулей'):
    m.inject_text(text);q=next(q for q in questions(g) if q['sentence']==text);a,b=text.casefold().split(' состоит из ')
    result=correct(g,q['id'],'incorrect',b,'PART_OF',a,source='train:mechanics' if corrections==0 else 'validation:instruments');corrections+=1;mutations+=result['accepted']
  if mode=='imitation':s['rules']=initial;mutations=0  # Correction records alone do not alter a rule.
  tp=fp=fn=unknown=0
  for sentence,a,r,b in CASES:
   if mode=='baseline':
    p=parse_definition(sentence);pred={(p['subject'],p['relation'],p['target'])} if p.get('ok') else set()
   else:
    fs=interpret(sentence,s['rules']);pred={(f['subject'],f['relation'],f['object']) for f in fs if not f['negated'] and f['status']=='observed'}
   gold={(a,r,b)} if a else set();tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred);unknown+=not pred
  peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop();precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
  return dict(mode=mode,seed=seed,tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=2*precision*recall/max(1e-9,precision+recall),unknown=unknown,corrections=corrections,accepted_mutations=mutations,seconds=time.perf_counter()-start,peak_bytes=peak)
with contextlib.redirect_stdout(io.StringIO()):rows=[run(mode,seed) for seed in (17,29,43) for mode in ('baseline','fixed','imitation','mutations','active')]
print(json.dumps(dict(cases=CASES,results=rows,note='imitation stores corrections without applying rule changes; mutations/active use identical labels, no additional gain is claimed for question ordering; baseline is legacy nominal extraction, not every internal causal hypothesis'),ensure_ascii=False,indent=2))
