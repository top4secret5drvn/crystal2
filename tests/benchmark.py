"""Synthetic A-D benchmark; run from repo root, optionally --source BASELINE_DIR."""
import argparse, contextlib, io, json, random, sys, time, tracemalloc
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--source',default=str(Path(__file__).resolve().parents[1]));args=p.parse_args()
sys.path.insert(0,args.source)
from cognitive_modules import ConceptFormationEngine, BeliefRevisionEngine, Reason
from engine import CrystalLattice, EDGE_SYNTAGM, EDGE_CAUSE, unpack_edge
random.seed(1729)

def experiment():
    # Disjoint IDs and independently assigned labels; target labels never enter
    # inference. One genuine rule and one training-only spurious correlation.
    facts={}
    for group in ('clean','spurious'):
        for i in range(24):
            for pred in (('a','b') if group=='clean' else ('d','e')):
                facts[f'{pred}({group}{i})']=(1,Reason('input'))
            facts[f'c({group}{i})']=(1 if group=='clean' or i<18 else -1,Reason('input'))
    kb=SimpleNamespace(facts=facts,rules=[])
    engine=ConceptFormationEngine(kb);engine.analyze_facts(min_confidence=.7)
    candidates=engine.discovered_concepts
    # Fixed predictions frozen before observing validation labels.
    samples=[({'a','b'},True) for _ in range(20)]+[({'d','e'},False) for _ in range(20)]
    random.shuffle(samples)
    validation,test=samples[:20],samples[20:]
    raw=[c for c in candidates if c['consequent']=='c']
    validated=[c for c in raw if sum(set(c['pattern'])<=x for x,y in validation)>=2
               and all(y for x,y in validation if set(c['pattern'])<=x)]
    # Exercise the lattice hypothesis lifecycle, not only an external filter.
    if hasattr(CrystalLattice, 'evaluate_abstraction'):
        lattice=CrystalLattice()
        validated=[]
        for c in raw:
            r=lattice.get_or_create('mdl:'+'+'.join(c['pattern']))
            from engine import CrystalReason
            lattice.add_hypothesis(r.id,CrystalReason('hypothesis',metadata={'training_examples':c['objects']}))
            for i,(x,y) in enumerate(validation):
                if set(c['pattern']) <= x:
                    lattice.evaluate_abstraction(r.id,f'validation-{i}',True,y,independent=True)
            if r.belief_status=='confirmed': validated.append(c)
    metrics={}
    for arm,concepts in [('disabled',[]),('unvalidated',raw),('heldout_validated',validated)]:
        arm_start=time.perf_counter()
        tracemalloc.reset_peak()
        predictions=[any(set(c['pattern'])<=x for c in concepts) for x,y in test]
        metrics[arm]={'accuracy':sum(pred==y for pred,(x,y) in zip(predictions,test))/len(test),
                      'false_positives':sum(pred and not y for pred,(x,y) in zip(predictions,test)),
                      'test_count':len(test),'rules':len(concepts),
                      'seconds':time.perf_counter()-arm_start,
                      'peak_bytes':tracemalloc.get_traced_memory()[1]}
    e=BeliefRevisionEngine()
    e.add_belief('p',1,Reason('input',confidence=.6,metadata={'источник':'a'}))
    e.add_belief('p',-1,Reason('input',confidence=.9,metadata={'источник':'b'}))
    e.set_source_reliability('b',.1)
    g=CrystalLattice();g.connect('rain','umbrella',weight=90,edge_type=EDGE_SYNTAGM)
    for r in g.resonators.values():r.energy=900
    g.infer_causal_links()
    a,b=g.label_to_id['rain'],g.label_to_id['umbrella']
    g.connect('heater','warmth',weight=90,edge_type=EDGE_CAUSE)
    return {'A_D':metrics,'B_restored':e.query('p')[0]==1,
            'C_spurious_cause':unpack_edge(g.resonators[a].connections[b])[1]==EDGE_CAUSE,
            'C_explicit_cause':unpack_edge(g.resonators[g.label_to_id['heater']].connections[g.label_to_id['warmth']])[1]==EDGE_CAUSE}
tracemalloc.start();start=time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()): result=experiment()
result.update(seed=1729,seconds=time.perf_counter()-start,peak_bytes=tracemalloc.get_traced_memory()[1])
print(json.dumps(result,indent=2))
