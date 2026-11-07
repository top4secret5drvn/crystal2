"""Deterministic held-out grammar/negative matrix; no label-dependent optimization."""
import contextlib,io,json,sys,tempfile,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args();sys.path.insert(0,a.source)
from engine import CrystalLattice
from membrane import LanguageMembrane
from research import ingest_observation

CASES=[
 ('Яблоко — это фрукт.','яблоко','фрукт'),
 ('Яблоко является плодом яблони.','яблоко','плод яблони'),
 ('Апельсин — цитрусовый плод.','апельсин','цитрусовый плод'),
 ('Сервер — это вычислительный узел.','сервер','вычислительный узел'),
 ('Рубин является минералом.','рубин','минерал'),
 ('Фотон является квантом излучения.','фотон','квант излучения'),
 ('Книга — печатное издание.','книга','печатное издание'),
 ('Процессор — функциональный блок.','процессор','функциональный блок'),
 ('Вода — холодная.','вода',None),
 ('Яблоко происходит от яблони.','яблоко',None),
 ('Пакет принадлежит пользователю.','пакет',None),
 ('Ключ растёт на дереве.','ключ',None),
 ('Яблоко не является минералом.','яблоко',None),
 ('Возможно яблоко это минерал.','яблоко',None),
 ('Объект это предмет это система.','объект',None),
 ('Солнце нагревает камень.','солнце',None),
]
rows=[]
with contextlib.redirect_stdout(io.StringIO()):
 for sentence,query,target in CASES:
  with tempfile.TemporaryDirectory() as d:
   g=CrystalLattice();m=LanguageMembrane(g,data_dir=d);ingest_observation(m,g,sentence,'fixture:'+query)
   if hasattr(m,'respond'):
    r=m.respond(query);v=m.respond(query,'verified')
    got=r['relations'][0]['object'] if r['relations'] else None
    if got and got.startswith('root:'):got=m._get_surface_form(got)
    mode=r['mode'];verified=v['mode']=='Verified'
   else:
    text=m.generate_text([query]);mode='Unknown' if 'Недостаточно' in text else 'Verified';got=None;verified=mode=='Verified'
   correct=(mode=='Attributed' and got==target) if target else mode=='Unknown'
   rows.append(dict(sentence=sentence,expected_target=target,mode=mode,actual_target=got,correct=correct,verified=verified))
print(json.dumps({'rows':rows,'supported_correct':sum(r['correct'] for r in rows if r['expected_target']),
 'supported_count':sum(r['expected_target'] is not None for r in rows),'unsupported_safe':sum(r['correct'] for r in rows if not r['expected_target']),
 'unsupported_count':sum(r['expected_target'] is None for r in rows),'false_verified':sum(r['verified'] for r in rows)},ensure_ascii=False,indent=2))
