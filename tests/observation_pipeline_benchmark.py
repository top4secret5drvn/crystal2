"""Offline chain reproduction; --source runs against a detached source directory."""
import argparse,contextlib,io,json,sys,tempfile,random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',default=str(Path(__file__).resolve().parents[1]));args=p.parse_args()
sys.path.insert(0,args.source)
from engine import CrystalLattice,CrystalPopulation,EDGE_NAMES,unpack_edge
from membrane import LanguageMembrane
from command_session import CommandSession
from persistence import CrystalSnapshot
from research import research_steps,ingest_observation
from simple_imitation import prepare
from imitation_learning import ImitationLearner,Config

ARTICLE='Яблоко — это фрукт. Яблоко является плодом яблони. Апельсин — цитрусовый плод.'

def run(legacy=None):
    random.seed(17)
    with tempfile.TemporaryDirectory() as d:
        g=CrystalLattice()
        if legacy:assert CrystalSnapshot.load(g,str(Path(args.source)/legacy))
        pop=CrystalPopulation();pop.crystals=[g]
        membrane=LanguageMembrane(g,data_dir=d);session=CommandSession(pop,membrane);rows=[]
        def collect(step):
            types={}
            for r in g.resonators.values():
                for _,packed in r.connections.typed_items():
                    name=EDGE_NAMES[unpack_edge(packed)[1]];types[name]=types.get(name,0)+1
            verified=sum(r.metadata.get('status')=='confirmed' for k,r in g.edge_meta.items() if len(k)==3)
            if hasattr(membrane,'respond'):
                result=membrane.respond('яблоко');mode=result['mode'];text=result['text'];admissible=int(mode in ('Verified','Attributed','Hypothesis'))
            else:
                text=membrane.generate_text(['яблоко']);admissible=int('Недостаточно' not in text);mode='Verified' if admissible else 'Unknown'
            rows.append(dict(step=step,nodes=len(g.resonators),observations=len(g.relation_observations),typed_relations=types,
                hypotheses=len(g.hypotheses)+len(g.causal_hypotheses),confirmed=verified,admissible_answers=admissible,mode=mode,text=text))
        collect('initial')
        for command in ('ввод яблоко это фрукт','спроси яблоко','ответь'):
            result=session.execute(command);assert not result.error,result.output;collect(command)
        provider=lambda topic:dict(text=ARTICLE,source='offline:Wikipedia-contract-fixture',source_type='wiki',title=topic)
        for _ in research_steps(membrane,g,'яблоко',ticks=1,fetcher=provider):pass
        collect('изучи яблоко (offline fixture)')
        session.execute('спроси яблоко');collect('спроси яблоко after study');session.execute('ответь');collect('ответь after study')
        e,controls,analysis=prepare(session,'Что такое яблоко?','Яблоко — это плод яблони.')
        learner=ImitationLearner(g,controls);record=learner.train_on_example(e,Config(enabled=True))
        collect('imitation before explicit evidence')
        initial_imitation=dict(reliable=analysis['reliable'],accepted=record['accepted'],reason=record['reason'])
        for sentence in ('Яблоко — это плод яблони.','Апельсин — это цитрусовый плод.'):
            ingest_observation(membrane,g,sentence,'explicit user demonstration','user_correction')
            collect('explicit user witness: '+sentence)
        e,controls,analysis=prepare(session,'Что такое яблоко?','Яблоко — это плод яблони.')
        learner=ImitationLearner(g,controls)
        flags=dict(enabled=True)
        if 'allow_user_observed' in Config.__dataclass_fields__:flags['allow_user_observed']=True
        calibrated=learner.train_on_example(e,Config(**flags))
        collect('calibration after explicit evidence')
        return dict(legacy=legacy,steps=rows,imitation=initial_imitation,after_witness=dict(accepted=calibrated['accepted'],reason=calibrated['reason'],changed=calibrated.get('changed',{}),
            baseline_metrics=calibrated.get('baseline_metrics'),candidate_metrics=[a.get('metrics') for a in calibrated.get('attempts',[])]))
with contextlib.redirect_stdout(io.StringIO()):results=[run(),run('bugfix_test.cry'),run('bugfix2.cry')]
print(json.dumps(results,ensure_ascii=False,indent=2))
