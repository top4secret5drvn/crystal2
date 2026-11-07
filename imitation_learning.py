"""Optional bounded local calibration of a traced, read-only relation selector.
No text-pair lookup, graph mutation, LLM, or universal Russian semantic grader.
"""
from dataclasses import dataclass, field, asdict, replace
import copy
import json
import math
import random
import re


@dataclass(frozen=True)
class TrainingExample:
    id: str
    query: str
    references: tuple
    relations: tuple = ()  # acceptable (subject label, EDGE_NAMES type, target label)
    context: str = 'global'
    world: str = ''
    concept_ids: tuple = ()
    correction_source: str = 'unspecified'
    correction_confidence: float = 0.0
    error_label: str = ''
    split: str = 'train'
    evidence_mode: str = 'verified'

    def __post_init__(self):
        for name in ('references','concept_ids'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        object.__setattr__(self,'relations',tuple(tuple(r) for r in self.relations))
        if self.evidence_mode not in ('verified','user_observed'):raise ValueError('Invalid evidence mode')
        if not self.id or self.split not in ('train','validation','test'):
            raise ValueError('Unique ID and explicit split required')
        if not math.isfinite(self.correction_confidence) or not 0<=self.correction_confidence<=1:
            raise ValueError('Invalid correction confidence')


@dataclass(frozen=True)
class Parameters:
    request_weight: float = 0.0
    is_a_form: int = 0
    isa_priority: float = 0.0
    part_priority: float = 0.0
    cause_priority: float = 0.0
    is_a_construction: int = 0

    def __post_init__(self):
        if self.is_a_form not in (0,1) or self.is_a_construction not in (0,1) or any(not math.isfinite(x) or not -2<=x<=2 for x in
            (self.request_weight,self.isa_priority,self.part_priority,self.cause_priority)):
            raise ValueError('Parameter out of bounded range')


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    seed: int = 17
    max_candidates: int = 12
    max_evaluations: int = 300
    guard_previous: bool = True
    min_confidence: float = .8
    allow_user_observed: bool = False

    def __post_init__(self):
        if (not isinstance(self.max_candidates,int) or not 0<=self.max_candidates<=64 or
            not isinstance(self.max_evaluations,int) or not 0<=self.max_evaluations<=4096 or
            not math.isfinite(self.min_confidence) or not 0<=self.min_confidence<=1):
            raise ValueError('Invalid bounded configuration')


@dataclass
class Attempt:
    text: str
    relation: tuple = ()
    path: tuple = ()
    confidence: float = 0.0
    alternatives: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    evidence_status: str = 'unknown'


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def instrumental(noun):
    # Deliberately restricted regular singular grammar, no lookup of learned pairs.
    if noun.endswith('а'): return noun[:-1]+'ой'
    if noun and noun[-1] in 'бвгджзклмнпрстфхцчшщ': return noun+'ом'
    return noun


def answer(lattice, example, parameters=Parameters()):
    from structured_observations import context_applies
    from engine import EDGE_NAMES, unpack_edge
    if example.world and (lattice.semantic_space is None or example.world not in lattice.semantic_space.worlds):
        return Attempt('Неизвестный смысловой мир.',parameters=asdict(parameters))
    parts=example.query.split(maxsplit=1)
    requested=parts[0] if parts and parts[0] in ('IS_A','PART_OF','CAUSE','SYN') else ''
    label=parts[1] if requested and len(parts)>1 else example.query
    rid=lattice.label_to_id.get(label)
    candidates=[]
    priorities={'IS_A':parameters.isa_priority,'PART_OF':parameters.part_priority,'CAUSE':parameters.cause_priority}
    if rid is not None:
        r=lattice.resonators[rid]
        for index,(target,packed) in enumerate(r.connections.typed_items()):
            if index>=4096:return Attempt("Бюджет обхода исчерпан.",parameters=asdict(parameters))
            weight,kind=unpack_edge(packed);relation=EDGE_NAMES[kind]
            reason=lattice.edge_meta.get((rid,target,kind))
            if (relation not in priorities or target not in lattice.resonators or reason is None or
                reason.kind=='hypothesis' or not (reason.metadata.get('status')=='confirmed' or
                (example.evidence_mode=='user_observed' and reason.metadata.get('status')=='user_observed')) or
                not (reason.context in ('global',example.context) or (example.evidence_mode=='user_observed' and reason.metadata.get('status')=='user_observed' and context_applies(lattice,reason,example.context,True)))):
                continue
            if example.world:
                bindings=lattice.semantic_bindings
                if any(len(bindings.get(label,[]))!=1 or example.world not in lattice.semantic_space.senses[bindings[label][0]].worlds
                       for label in (r.label,lattice.resonators[target].label)):
                    continue
            confidence=min(reason.confidence,.6) if reason.metadata.get('status')=='user_observed' else reason.confidence
            if not math.isfinite(confidence) or not 0<confidence<=1:continue
            target_label=lattice.resonators[target].label
            score=weight/100+priorities[relation]+parameters.request_weight*(relation==requested)
            candidates.append((score,(label,relation,target_label),(rid,target),confidence,dict(
                source=reason.source_label,source_type=reason.source_type,context=reason.context,timestamp=reason.timestamp,metadata=copy.deepcopy(reason.metadata))))
    candidates.sort(key=lambda c:(-c[0],c[1]))
    trace=Attempt('Недостаточно подтвержденных оснований.',parameters=asdict(parameters),
                  alternatives=[c[1] for c in candidates])
    if not candidates:return trace
    if len(candidates)>1 and abs(candidates[0][0]-candidates[1][0])<1e-12:
        trace.text='Неоднозначность: несколько равноприоритетных отношений.'
        return trace
    _,relation,path,confidence,provenance=candidates[0]
    subject,kind,target=relation
    from structured_observations import surface_label
    subject_text=surface_label(lattice,subject);target_text=surface_label(lattice,target)
    if kind=='IS_A' and parameters.is_a_construction:
        text=f'{subject_text.capitalize()} — это {target_text}.'
    elif kind=='IS_A': text=f'{subject_text.capitalize()} является {instrumental(target_text) if parameters.is_a_form else target_text}.'
    elif kind=='PART_OF': text=f'{subject.capitalize()} входит в состав {target}.'
    else:text=f'{subject.capitalize()} вызывает {target}.'
    return Attempt(text,relation,path,confidence,trace.alternatives,asdict(parameters),[provenance],provenance['metadata'].get('status','unknown'))


def evaluate(attempt, example):
    known=bool(example.relations)
    semantic=float(attempt.relation in example.relations) if known else None
    relation_type=float(bool(attempt.relation) and any(attempt.relation[1]==r[1] for r in example.relations)) if known else None
    concepts=float(bool(attempt.relation) and any((attempt.relation[0],attempt.relation[2])==(r[0],r[2]) for r in example.relations)) if known else None
    form=float(any(normalize(attempt.text)==normalize(ref) for ref in example.references)) if example.references else None
    confidence_quality=1-abs(attempt.confidence-(semantic if semantic is not None else 0)) if known else None
    # No semantic gold => no fabricated optimization target.
    score=None if not known else .6*semantic+.15*relation_type+.1*concepts+.1*(form or 0)+.05*confidence_quality
    return dict(semantic=semantic,relation_type=relation_type,concepts=concepts,form=form,
                confidence_quality=confidence_quality,score=score)


def diagnose(attempt, example):
    gold=set(example.relations)
    available=set(attempt.alternatives)
    if gold and not gold & available:return {'kind':'D','alternatives':['missing_or_unconfirmed_knowledge']}
    if not gold:return {'kind':'E' if len(available)>1 else 'unknown','alternatives':['A','B','C','D','E']}
    if not attempt.relation:return {'kind':'E','alternatives':['ambiguous_selection']}
    if attempt.relation not in gold:
        return {'kind': 'B' if any((attempt.relation[0],attempt.relation[2])==(r[0],r[2]) for r in gold) else 'C', 'alternatives':['C' if attempt.relation[1] not in {r[1] for r in gold} else 'selection_error']}
    if evaluate(attempt,example)['form']==0:return {'kind':'A','alternatives':[]}
    return {'kind':'correct','alternatives':[]}


class TrainingCancelled(RuntimeError):pass


class ImitationLearner:
    def __init__(self,lattice,validation=(),state=None):
        if any(e.split!='validation' for e in validation):raise ValueError('Validation split required')
        if len({e.id for e in validation})!=len(validation):raise ValueError('Duplicate validation IDs')
        self.lattice=lattice;self.validation=tuple(validation)
        self.parameters=Parameters();self.history=[];self.previous=[];self.skills=[]
        if state:self.restore(state)

    def restore(self,state):
        state=copy.deepcopy(state)
        if state['schema']!=1:raise ValueError('Unsupported imitation state')
        self.parameters=Parameters(**state['parameters']);self.history=list(state['history'])
        self.previous=[Parameters(**p) for p in state['previous']]
        self.skills=[TrainingExample(**{**e,'relations':tuple(tuple(r) for r in e['relations'])}) for e in state['skills']]

    def state(self):
        return copy.deepcopy(dict(schema=1,parameters=asdict(self.parameters),history=self.history,
                    previous=[asdict(p) for p in self.previous],skills=[asdict(e) for e in self.skills]))

    def rollback(self):
        if not self.previous:return False
        self.parameters=self.previous.pop()
        self.history.append({'action':'manual_rollback','parameters':asdict(self.parameters)})
        return True

    def train_on_example(self,example,config=Config(),*,on_event=None,should_stop=lambda:False):
        if example.split!='train':raise ValueError('Only train examples may generate mutations')
        if any(e.id==example.id or e.query==example.query for e in self.validation):raise ValueError('Train/validation overlap')
        if config.max_candidates<0 or config.max_evaluations<0:raise ValueError('Invalid budget')
        before=answer(self.lattice,example,self.parameters);diagnosis=diagnose(before,example)
        record=dict(id=example.id,query=example.query,correction_confidence=example.correction_confidence,
                    annotated_relations=list(example.relations),concept_ids=list(example.concept_ids),source=example.correction_source,scope={'context':example.context,'world':example.world,'evidence_mode':example.evidence_mode},
                    before=asdict(before),references=list(example.references),diagnosis=diagnosis,human_error_label=example.error_label,
                    accepted=False,attempts=[],evaluations=0,reason='not_enabled')
        if on_event:on_event(dict(stage='initial_attempt',attempt=asdict(before),diagnosis=diagnosis,comparison=evaluate(before,example)))
        if should_stop():raise TrainingCancelled('Training cancelled')
        if not config.enabled: return record
        if (example.evidence_mode=='user_observed' or any(e.evidence_mode=='user_observed' for e in self.validation)) and not config.allow_user_observed:
            record['reason']='user_observed_policy_not_enabled';self.history.append(record);return record
        if any(e.id==example.id for e in self.skills) and evaluate(before,example)['semantic']==1 and evaluate(before,example)['form']!=0:
            record['reason']='already_consolidated';return record
        if (not self.validation or any(not e.relations or e.correction_confidence<config.min_confidence for e in self.validation) or
            example.correction_confidence<config.min_confidence or not example.relations or diagnosis['kind']=='D'):
            record['reason']='missing_validation_or_evidence';self.history.append(record);return record
        controls=list(self.validation)+(self.skills if config.guard_previous else [])
        suite=[example]+controls
        if len(suite)>config.max_evaluations:
            record['reason']='budget';self.history.append(record);return record
        last_metrics=[];last_train_attempt=None
        def scores(params):
            nonlocal last_metrics,last_train_attempt
            record['evaluations']+=len(suite)
            last_train_attempt=answer(self.lattice,suite[0],params)
            last_metrics=[evaluate(last_train_attempt,suite[0])]+[evaluate(answer(self.lattice,e,params),e) for e in suite[1:]]
            return [m['score'] for m in last_metrics]
        baseline=scores(self.parameters)
        baseline_metrics=last_metrics
        record['baseline_metrics']=baseline_metrics
        # Proposals depend on train diagnosis and fixed bounded steps, never on control answers.
        candidates=[]
        if diagnosis['kind']=='A':candidates=[replace(self.parameters,is_a_form=1),replace(self.parameters,is_a_construction=1)]
        else:
            for name in ('request_weight','isa_priority','part_priority','cause_priority'):
                for step in (-.25,.25):
                    value=getattr(self.parameters,name)+step
                    if -2<=value<=2:candidates.append(replace(self.parameters,**{name:value}))
        random.Random(config.seed).shuffle(candidates)
        best=None;best_score=baseline[0]
        for candidate in candidates[:config.max_candidates]:
            if should_stop():raise TrainingCancelled('Training cancelled before next candidate')
            if record['evaluations']+len(suite)>config.max_evaluations:break
            result=scores(candidate)
            valid=calibration_gain(baseline,result)
            valid=valid and all(after[k]>=before[k]-1e-12 for before,after in zip(baseline_metrics[1:],last_metrics[1:])
                                for k in ('semantic','relation_type','concepts','form','confidence_quality')
                                if before[k] is not None and after[k] is not None)
            record['attempts'].append(dict(parameters=asdict(candidate),train=result[0],controls=result[1:],metrics=last_metrics,accepted=valid))
            if on_event:on_event(dict(stage='candidate',iteration=len(record['attempts']),attempt=asdict(last_train_attempt),
                diagnosis=diagnose(last_train_attempt,example),comparison=last_metrics[0],controls=last_metrics[1:],
                eligible=valid,parameters=asdict(candidate),changed={k:v for k,v in asdict(candidate).items() if v!=getattr(self.parameters,k)}))
            if should_stop():raise TrainingCancelled('Training cancelled after current evaluation')
            if valid and result[0]>best_score:
                best,best_score=candidate,result[0]
        if should_stop():raise TrainingCancelled('Training cancelled before parameter application')
        if best is not None:
            self.previous.append(self.parameters);self.parameters=best
            if not any(e.id==example.id for e in self.skills):self.skills.append(example)
            record.update(accepted=True,reason='train_gain_without_control_regression')
        else:record['reason']='no_supported_improvement'
        record['after']=asdict(answer(self.lattice,example,self.parameters))
        record['baseline_scores']=baseline
        record['changed']={k:[getattr(self.previous[-1],k),v] for k,v in asdict(self.parameters).items()
                           if best is not None and getattr(self.previous[-1],k)!=v}
        self.history.append(record)
        return record


def cli():
    import argparse
    from engine import CrystalLattice
    from persistence import CrystalSnapshot
    parser=argparse.ArgumentParser(description='Optional imitation calibration session')
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--examples',required=True,help='JSON TrainingExample list, with explicit splits')
    args=parser.parse_args()
    lattice=CrystalLattice()
    if not CrystalSnapshot.load(lattice,args.snapshot):raise SystemExit('Snapshot failed to load')
    examples=[TrainingExample(**row) for row in json.loads(open(args.examples,encoding='utf8').read())]
    if len({e.id for e in examples})!=len(examples):raise SystemExit('Duplicate example IDs')
    learner=ImitationLearner(lattice,[e for e in examples if e.split=='validation'],getattr(lattice,'imitation_state',None))
    selected=None;enabled=False;last=None
    print('калибровка старт | пример <train ID> | эталон <текст> | обучи | отчёт | откат | сохрани <новый файл> | выход')
    while True:
        try:line=input('КАЛИБРОВКА> ').strip()
        except EOFError:break
        if line.startswith('калибровка '):line=line[len('калибровка '):]
        command,_,value=line.partition(' ')
        if command=='выход':break
        if command=='старт':enabled=True;print('Опциональная калибровка включена; консолидация во сне выключена.')
        elif command=='пример':
            selected=next((e for e in examples if e.id==value and e.split=='train'),None)
            print(asdict(answer(lattice,selected,learner.parameters)) if selected else 'Не найден train ID')
        elif command=='эталон' and selected:
            # Text alone never adds facts or semantic gold.
            selected=replace(selected,references=(value,))
        elif command=='обучи' and selected:
            last=learner.train_on_example(selected,Config(enabled=enabled));print(json.dumps(last,ensure_ascii=False))
        elif command=='отчёт':print(json.dumps(last or learner.state(),ensure_ascii=False,indent=2))
        elif command=='откат':print(learner.rollback())
        elif command=='сохрани' and value:
            from pathlib import Path
            if Path(value).exists():print('Используйте новый файл; исходный snapshot сохраняется.');continue
            lattice.imitation_state=learner.state();CrystalSnapshot.save(lattice,value)
        else:print('Неизвестная команда или не выбран пример.')



def calibration_gain(before,after):
    """Shared acceptance contract for generator and interpreter candidate suites."""
    return bool(after and after[0]>before[0]+1e-12 and all(a>=b-1e-12 for b,a in zip(before[1:],after[1:])))


def calibrate_interpretation(lattice,example,rule_id,max_candidates=24):
    """Local rule candidates use the same guarded train/control acceptance contract."""
    import copy
    from semantic_rules import state,interpret,validate_rules,REGISTRY
    s=state(lattice);controls=[c for c in s['controls'] if c['sentence']!=example['sentence']]
    if len(controls)>32:raise ValueError('Calibration budget: 32 controls')
    def score(rules,e):
        frames=interpret(e['sentence'],rules,e.get('context','global'))
        return float(len(frames)==1 and [frames[0]['subject'],frames[0]['relation'],frames[0]['object']]==e['gold'] and not frames[0]['negated'])
    suite=[example]+controls;before=[score(s['rules'],e) for e in suite]
    result=dict(accepted=False,reason='missing_independent_controls',before=before,attempts=[],new_facts=0)
    if not controls:return result
    candidates=[]
    for index,r in enumerate(s['rules']):
        if r['id']!=rule_id and r['marker'] not in example['sentence']:continue
        for field,values in [('reverse',[False,True]),('relation',[example['gold'][1]]),('priority',[min(5,r['priority']+1)]),('confidence',[.6,.8]),('context',[example.get('context','global')])]:
            for value in values:
                if value==r.get(field):continue
                rules=copy.deepcopy(s['rules']);rules[index][field]=value;rules[index]['version']+=1;rules[index]['source']=example['source']
                candidates.append((rules,dict(rule=r['id'],field=field,value=value)))
    best=None;best_score=before[0]
    for rules,change in candidates[:min(24,max_candidates)]:
        validate_rules(rules);after=[score(rules,e) for e in suite];valid=calibration_gain(before,after)
        result['attempts'].append(dict(change=change,scores=after,eligible=valid))
        if valid and after[0]>best_score:best=(rules,change,after);best_score=after[0]
    if best:
        s['previous'].append(copy.deepcopy(s['rules']));s['previous']=s['previous'][-20:]
        s['rules'],change,after=best;s['version']+=1
        result.update(accepted=True,reason='train_gain_without_control_regression',change=change,after=after)
    else:result['reason']='no_supported_improvement'
    return result

if __name__=='__main__':cli()
