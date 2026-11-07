"""Offline fixed-budget transfer and sequential-forgetting experiments."""
import contextlib,io,json,random,sys,time,tracemalloc
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import CrystalLattice,CrystalReason,EDGE_IS_A,EDGE_PART_OF,EDGE_CAUSE
from imitation_learning import *


def build(seed):
    rng=random.Random(seed);g=CrystalLattice();examples=[]
    for split,names in [('train',['яблоко','молоток','модуль']),('validation',['апельсин','прибор','блок']),('test',['банан','станок','узел'])]:
        noun_sets={'train':['предмет','объект','инструмент'], 'validation':['элемент','материал','организм'], 'test':['металл','ресурс','компонент']}
        for index,name in enumerate(names):
            for kind,target,weight in [('IS_A',noun_sets[split][index],70),('PART_OF','система',90),('CAUSE','тепло',75)]:
                edge={'IS_A':EDGE_IS_A,'PART_OF':EDGE_PART_OF,'CAUSE':EDGE_CAUSE}[kind]
                g.connect(name,target,weight+rng.randint(-1,1),edge,CrystalReason('observation',confidence=.9,metadata={'status':'confirmed'}))
                references=(f'{name.capitalize()} является {instrumental(target)}.',) if kind=='IS_A' else ()
                examples.append(TrainingExample(split+name+kind,kind+' '+name,references,((name,kind,target),),
                    correction_source='controlled fixture',correction_confidence=1,split=split))
    return g,examples


def quality(g,examples,params,memory=None):
    results=[]
    for e in examples:
        a=answer(g,e,params)
        if memory is not None and e.query in memory:a=memory[e.query]
        results.append(evaluate(a,e))
    return {'semantic':sum(r['semantic'] for r in results)/len(results),
        'form':sum(r['form'] for r in results if r['form'] is not None)/max(1,sum(r['form'] is not None for r in results)),
        'confident_errors':sum(r['semantic']==0 and answer(g,e,params).confidence>=.8 for e,r in zip(examples,results))}


def run(seed,mode):
    start=time.perf_counter();tracemalloc.start();g,examples=build(seed)
    train=[e for e in examples if e.split=='train'];val=[e for e in examples if e.split=='validation'];test=[e for e in examples if e.split=='test']
    learner=ImitationLearner(g,val);memory={}
    for epoch in range(2):
        for e in train:
            if mode=='memorization':memory[e.query]=Attempt(e.references[0] if e.references else answer(g,e).text,e.relations[0],confidence=.9)
            elif mode!='original':learner.train_on_example(e,Config(enabled=True,seed=seed,guard_previous=mode=='guarded'))
    train_score=quality(g,train,learner.parameters,memory if mode=='memorization' else None)
    test_score=quality(g,test,learner.parameters,memory if mode=='memorization' else None)
    result=dict(seed=seed,mode=mode,train=train_score,test=test_score,gap=train_score['semantic']-test_score['semantic'],
        accepted=sum(r.get('accepted',False) for r in learner.history),
        rollbacks=sum(not c['accepted'] for r in learner.history for c in r.get('attempts',[])),
        candidates=sum(len(r.get('attempts',[])) for r in learner.history),
        evaluations=sum(r.get('evaluations',0) for r in learner.history),
        first_gain_candidate=next((i+1 for r in learner.history for i,c in enumerate(r.get('attempts',[])) if c['accepted']),None),
        seconds=time.perf_counter()-start,peak_bytes=tracemalloc.get_traced_memory()[1])
    tracemalloc.stop();return result


def forgetting(seed,guard):
    g,examples=build(seed)
    first=replace(examples[0],id='stage1',query='яблоко')
    oldtest=replace(next(e for e in examples if e.query=='IS_A банан'),query='банан')
    v1=replace(next(e for e in examples if e.query=='IS_A апельсин'),query='апельсин')
    l=ImitationLearner(g,[v1]);cfg=Config(enabled=True,seed=seed,guard_previous=guard)
    for _ in range(2):l.train_on_example(first,cfg)
    before=evaluate(answer(g,oldtest,l.parameters),oldtest)['semantic']
    second=replace(next(e for e in examples if e.query=='CAUSE молоток'),id='stage2',query='молоток')
    v2=replace(next(e for e in examples if e.query=='CAUSE прибор'),query='прибор');l.validation=(v2,)
    record=l.train_on_example(second,cfg)
    after=evaluate(answer(g,oldtest,l.parameters),oldtest)['semantic']
    return dict(seed=seed,guard=guard,old_before=before,old_after=after,forgetting=before-after,new_task_accepted=record['accepted'])


def geometric_feedback(seed):
    from semantic_space import SemanticSpace,Evidence
    s=SemanticSpace(3,seed);s.add_world('objects',('entity','class'),'controlled geometry experiment')
    ids={x:s.add_sense(x,x,'class' if x in ('u','v') else 'entity',['objects']) for x in ('a','b','c','d','u','v')}
    for x,y in [('a','u'),('b','u'),('c','v'),('d','v')]:s.observe(Evidence(x,ids[x],ids[y],'IS_A','objects','fixture',confirmed=True))
    s.train('objects');baseline=SemanticSpace.from_dict(s.to_dict())
    # Validation chooses threshold; held-out pairs are never feedback constraints.
    validation=[('a','b',True),('a','c',False)]
    def score(model,a,b):return model.similarity(ids[a],ids[b],'objects') or 0
    threshold=max((.15,.3,.45),key=lambda t:sum((score(s,a,b)>=t)==y for a,b,y in validation))
    test=[('a','d',False),('b','c',False),('c','d',True)]
    before=sum((score(s,a,b)>=threshold)==y for a,b,y in test)/len(test)
    prediction=score(s,'a','c')>=threshold
    # External verified negative feedback, not self-generated truth. Disposable copy only.
    candidate=SemanticSpace.from_dict(s.to_dict())
    candidate.observe(Evidence('feedback',ids['a'],ids['c'],'INCOMPATIBLE','objects','independent experiment',confirmed=True))
    candidate.train('objects')
    after=sum((score(candidate,a,b)>=threshold)==y for a,b,y in test)/len(test)
    return dict(seed=seed,feedback_prediction=prediction,before=before,after=after,graph_accuracy=sum(bool(s.features(ids[a],'objects') & s.features(ids[b],'objects'))==y for a,b,y in test)/len(test),
        coordinate_shift=sum((x-y)**2 for x,y in zip(s.position(ids['a'],'objects'),candidate.position(ids['a'],'objects')))**.5,
        original_unchanged=s.to_dict()==baseline.to_dict(),committed=False)

if __name__=='__main__':
    rows=[run(seed,mode) for seed in (17,29,43) for mode in ('original','memorization','local','guarded')]
    print(json.dumps(dict(rows=rows,forgetting=[forgetting(seed,g) for seed in (17,29,43) for g in (False,True)],
        geometry=[geometric_feedback(seed) for seed in (17,29,43)]),indent=2))
