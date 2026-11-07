"""Offline transductive-input / held-out-target benchmark, fixed protocol.

No target label is ever placed in the geometry or feature graph. Validation
selects classification thresholds only. Test answers are read after prediction.
"""
import contextlib
import io
import json
import random
import sys
import time
import tracemalloc
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic_space import SemanticSpace, Evidence
from engine import CrystalLattice, CrystalReason, EDGE_IS_A
from persistence import CrystalSnapshot

SEEDS=(17,29,43)
K=3


def dataset(seed, corruption):
    rng=random.Random(seed)
    features={};groups={};labels={};split={}
    for group in range(8):
        # Hidden target independent of object IDs and of optimization.
        target=group%2
        for i in range(8):
            name=f'object-{group}-{i}'
            groups[name]=group;labels[name]=target
            split[name]='train' if i<4 else ('validation' if i<6 else 'test')
            f=set()
            for property_index in range(4):
                if rng.random() > (.55 if corruption=='incomplete' else .15):
                    f.add(f'feature-{group}-{property_index}')
            features[name]=f
    return features,groups,labels,split


def run(seed, dimensions, corruption='normal'):
    start=time.perf_counter();tracemalloc.start()
    features,groups,labels,split=dataset(seed,corruption)
    space=SemanticSpace(dimensions or 3,seed)
    space.add_world('objects',('entity','class'),'benchmark typed input')
    space.add_world('operators',('connector','function'),'cross-world probe')
    ids={name:space.add_sense(name,'object','entity',['objects']) for name in features}
    targets={f:space.add_sense(f,'feature','class',['objects']) for values in features.values() for f in values}
    g=CrystalLattice()
    g.configure_geometry(space,'objects','diagnostic' if dimensions else 'off')
    for name,sid in ids.items():
        g.get_or_create(name);g.bind_semantic_sense(name,sid)
    for f,sid in targets.items():g.get_or_create(f);g.bind_semantic_sense(f,sid)
    for name,values in features.items():
        for feature in sorted(values):
            reason=CrystalReason('observation',source_label='fixture:'+name+':'+feature,source_type='experiment',metadata={'status':'confirmed'})
            g.connect(name,feature,60,EDGE_IS_A,reason)
            space.observe(Evidence(name+':'+feature,ids[name],targets[feature],'IS_A','objects','fixture',confirmed=True))
            if corruption=='contradictory' and feature.endswith('-0'):
                reason.metadata.update(status='contradiction', counterevidence={'source':'independent-counterexample','positive':False,'confidence':1.0})
                space.observe(Evidence('negative:'+name+':'+feature,ids[name],targets[feature],'IS_A','objects','independent-counterexample',confirmed=True,positive=False))
    # Same resolved positive facts for graph and geometry; contradictions abstain.
    graph_features={name:space.features(sid,'objects') for name,sid in ids.items()}
    if dimensions:space.train('objects',iterations=120,max_operations=100000)
    train=sorted(n for n in ids if split[n]=='train')
    def nearest(name):
        scored=[]
        for other in train:
            if dimensions:
                score=space.similarity(ids[name],ids[other],'objects')
            else:
                a,b=graph_features[name],graph_features[other]
                score=len(a&b)/len(a|b) if a|b else None
            if score is not None and score>0:scored.append((score,other))
        return sorted(scored,key=lambda pair:(-pair[0],pair[1]))[:K]
    def probability(name):
        neighbors=nearest(name)
        if not neighbors:return None
        return sum(score*labels[other] for score,other in neighbors)/sum(score for score,_ in neighbors)
    validation=sorted(n for n in ids if split[n]=='validation')
    # Policy: unknown means abstention; not an observed false assertion.
    validation_scores={n:probability(n) for n in validation}
    thresholds=(.4,.5,.6)
    def predict(score,t):return None if score is None else score>=t
    threshold=max(thresholds,key=lambda t:(sum(predict(validation_scores[n],t)==bool(labels[n]) for n in validation),-abs(t-.5)))
    test=sorted(n for n in ids if split[n]=='test')
    predictions={n:predict(probability(n),threshold) for n in test}
    neighbor_results={n:nearest(n) for n in test}
    precision=sum(sum(groups[o]==groups[n] for _,o in neighbor_results[n])/K for n in test)/len(test)
    op=space.add_sense('однако','contrast','connector',['operators'])
    context_errors=sum(space.similarity(ids[n],op,'objects') is not None for n in test)
    g.defragment();size_after_first=len(g.resonators)
    for _ in range(4):g.defragment()
    additional_sleep_nodes=len(g.resonators)-size_after_first
    signature_list=[tuple(sorted(g._abstraction_leaves(node))) for node in g.abstraction_components]
    if not dimensions:
        g.configure_geometry(None,None,'off')
    with tempfile.TemporaryDirectory() as d:
        filename=str(Path(d)/'state.cry')
        CrystalSnapshot.save(g,filename)
        state_bytes=Path(filename).stat().st_size
    result={'seed':seed,'dimensions':dimensions or 'graph','condition':corruption,'K':K,
        'precision_at_k':precision,
        'hidden_relation_accuracy':sum(predictions[n]==bool(labels[n]) for n in test)/len(test),
        'false_positives':sum(predictions[n] is True and not labels[n] for n in test),
        'abstentions':sum(predictions[n] is None for n in test),
        'context_errors':context_errors,'duplicate_abstractions':len(signature_list)-len(set(signature_list)),
        'additional_sleep_nodes':additional_sleep_nodes,'threshold':threshold,
        'optimizer_operations':space.last_operations if dimensions else 0,
        'state_bytes':state_bytes,
        'seconds':time.perf_counter()-start,'peak_traced_bytes':tracemalloc.get_traced_memory()[1],
        'splits':{'train':len(train),'validation':len(validation),'test':len(test)}}
    tracemalloc.stop()
    return result


def main():
    rows=[]
    for condition in ('normal','incomplete','contradictory'):
        for seed in SEEDS:
            for dimensions in (None,2,3,8):
                with contextlib.redirect_stdout(io.StringIO()):row=run(seed,dimensions,condition)
                rows.append(row)
    print(json.dumps({'protocol':'typed-input transductive geometry; targets held out; 120 iterations / 100000 coordinate operations; K=3 fixed', 'rows':rows},indent=2))

if __name__=='__main__':main()
