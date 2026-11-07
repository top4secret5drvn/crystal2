"""Run identical input/query pairs against --source or current checkout."""
import argparse,contextlib,io,json,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args();sys.path.insert(0,a.source)
from engine import CrystalLattice
from membrane import LanguageMembrane
rows=[]
with contextlib.redirect_stdout(io.StringIO()):
 for sentence,query in [('кот ест рыбу','что ест кот?'),('кот ест рыбу','кто ест рыбу?'),('солнце это звезда','что такое солнце?'),('солнце нагревает землю','что нагревает землю?'),('мастер читает книгу','что читает мастер?'),('робот видит башню','кто видит башню?'),('кот ест рыбу','что строит кот?'),('кот не ест рыбу','что ест кот?')]:
  with tempfile.TemporaryDirectory() as d:
   g=CrystalLattice();m=LanguageMembrane(g,data_dir=d);stats=m.inject_text(sentence);r=m.respond(query)
   rows.append(dict(sentence=sentence,query=query,stats=vars(stats),response=r))
print(json.dumps(rows,ensure_ascii=False,indent=2))
