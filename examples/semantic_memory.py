"""Run: python3 examples/semantic_memory.py (offline)."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic_space import SemanticSpace, Evidence
from engine import CrystalLattice

s=SemanticSpace(3,17)
s.add_world('objects',('entity','class'),'demonstration only')
fruit=s.add_sense('фрукт','fruit','class',['objects'])
apple=s.add_sense('яблоко','apple','entity',['objects'])
orange=s.add_sense('апельсин','orange','entity',['objects'])
for sid in (apple,orange):
    s.observe(Evidence(sid,sid,fruit,'IS_A','objects','independent observation',confirmed=True))
s.train('objects');before=s.position(apple,'objects')
apricot=s.add_sense('абрикос','apricot','entity',['objects'])
s.observe(Evidence('new observation',apricot,fruit,'IS_A','objects','new observation',confirmed=True))
s.train('objects');after=s.position(apple,'objects')
g=CrystalLattice();g.configure_geometry(s,'objects','hypotheses')
for name,sid in [('яблоко',apple),('апельсин',orange),('абрикос',apricot)]:
    g.get_or_create(name);g.bind_semantic_sense(name,sid)
print(json.dumps({'before':before,'after':after,'coordinate_changed':before!=after,
    'comparison':g.geometry_comparison('яблоко','апельсин'),
    'hypotheses':g.rank_analogy_hypotheses('яблоко')},ensure_ascii=False,indent=2))
