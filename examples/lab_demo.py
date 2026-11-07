"""Create an explicitly synthetic, live-core Lab fixture. Never overwrite files.
python examples/lab_demo.py /path/to/lab-workspace
"""
import json,sys
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import CrystalLattice,CrystalReason,EDGE_IS_A,EDGE_SYNTAGM,EDGE_PART_OF
from persistence import CrystalSnapshot
from semantic_space import SemanticSpace,Evidence
from imitation_learning import TrainingExample
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
if (root/'lab-demo.cry').exists() or (root/'lab-examples.json').exists():raise SystemExit('Demo files already exist; use a new directory')
g=CrystalLattice();s=SemanticSpace(3,17);s.add_world('предметные понятия',('entity','class'),'Synthetic Lab demonstration')
roles={word:'entity' for word in ('яблоко','апельсин','абрикос','дерево','трава','куст')};roles.update({'фрукт':'class','растение':'class','предмет':'class','система':'class'})
ids={w:s.add_sense(w,w,role,['предметные понятия']) for w,role in roles.items()}
examples=[]
for i,word in enumerate(('яблоко','апельсин','абрикос','дерево','трава','куст')):
    target='фрукт' if i<3 else 'растение'
    reason=CrystalReason('observation',source_label='Synthetic Lab demonstration',confidence=.9,metadata={'status':'confirmed'})
    g.connect(word,target,70,EDGE_IS_A,reason);g.connect(word,target,20,EDGE_SYNTAGM)
    s.observe(Evidence(word,ids[word],ids[target],'IS_A','предметные понятия','Synthetic Lab demonstration',confirmed=True))
    g.get_or_create(word).energy=150+i*20
    if i<3:
        g.connect(word,'предмет',60,EDGE_IS_A,reason);g.connect(word,'система',90,EDGE_PART_OF,reason)
        examples.append(TrainingExample(word,'IS_A '+word,(word.capitalize()+' является фруктом.',),((word,'IS_A','фрукт'),),correction_source='Synthetic Lab demonstration',correction_confidence=1,split=('train','validation','test')[i]))
s.train('предметные понятия');g.configure_geometry(s,'предметные понятия','diagnostic')
for word,sid in ids.items():g.bind_semantic_sense(word,sid)
CrystalSnapshot.save(g,str(root/'lab-demo.cry'))
(root/'lab-examples.json').write_text(json.dumps([asdict(e) for e in examples],ensure_ascii=False,indent=2),encoding='utf8')
print('Load lab-demo.cry in Lab; import lab-examples.json in Calibration. This is synthetic test evidence.')
