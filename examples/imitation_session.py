"""Prepare an offline, explicitly synthetic interactive demonstration.
Usage: python3 examples/imitation_session.py /tmp/crystal-imitation-demo
"""
import json,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import CrystalLattice,CrystalReason,EDGE_IS_A,EDGE_PART_OF
from imitation_learning import TrainingExample
from persistence import CrystalSnapshot
out=Path(sys.argv[1]);out.mkdir(exist_ok=True)
if (out/'demo.cry').exists() or (out/'examples.json').exists():raise SystemExit('Use a new demo directory')
g=CrystalLattice();examples=[]
for name,split in [('яблоко','train'),('апельсин','validation'),('банан','test')]:
    for target,kind,weight in [('предмет',EDGE_IS_A,70),('система',EDGE_PART_OF,90)]:
        g.connect(name,target,weight,kind,CrystalReason('observation',source_label='synthetic demonstration',confidence=.9,metadata={'status':'confirmed'}))
    examples.append(TrainingExample(name,'IS_A '+name,(name.capitalize()+' является предметом.',),
        ((name,'IS_A','предмет'),),correction_source='demonstration',correction_confidence=1,split=split))
CrystalSnapshot.save(g,str(out/'demo.cry'))
(out/'examples.json').write_text(json.dumps([asdict(e) for e in examples],ensure_ascii=False,indent=2))
print('python3 imitation_learning.py --snapshot '+str(out/'demo.cry')+' --examples '+str(out/'examples.json'))
print('калибровка старт\nкалибровка пример яблоко\nкалибровка обучи\nкалибровка обучи\nкалибровка отчёт\nкалибровка откат\nвыход')
