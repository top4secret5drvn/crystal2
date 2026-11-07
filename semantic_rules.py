"""Bounded literal rule language: no regex supplied by users, no executable rules."""
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from engine import EDGE_NAMES,EDGE_CONSTRAINTS

REGISTRY={name:{'code':code,'directed':not EDGE_CONSTRAINTS[code].get('bidirectional',False),
    'evidence':'association' if name=='SYN' else 'linguistic_claim_not_independent_verification'} for code,name in EDGE_NAMES.items()}

@dataclass
class SemanticFrame:
    sentence:str
    subject:str
    object:str
    predicate:str
    relation:str
    direction:str='forward'
    negated:bool=False
    modality:str='asserted'
    conditions:str=''
    context:str='global'
    world:str=''
    source:str='direct text input'
    confidence:float=.65
    status:str='observed'
    rule_id:str=''
    alternatives:list=field(default_factory=list)
    id:str=''


def state(g):
    if getattr(g,'semantic_rules_state',None) is None:
        rules=json.loads(Path(__file__).with_name('semantic_rule_defaults.json').read_text(encoding='utf8'))
        g.semantic_rules_state={'schema':1,'version':1,'rules':rules,'frames':[], 'corrections':[], 'controls':[], 'previous':[], 'questions':{},'documents':[]}
    return g.semantic_rules_state


def validate_rules(rules):
    if not isinstance(rules,list) or len(rules)>64:raise ValueError('Rule budget: 64')
    ids=set()
    for r in rules:
        if set(r)-{'id','marker','relation','reverse','priority','confidence','context','version','source','success','failure'}:raise ValueError('Unsupported rule instruction')
        if not isinstance(r.get('id'),str) or r['id'] in ids:raise ValueError('Unique rule id required')
        ids.add(r['id'])
        if r.get('relation') not in REGISTRY:raise ValueError('Unregistered relation type')
        if not isinstance(r.get('marker'),str) or not 1<=len(r['marker'])<=80 or any(c in r['marker'] for c in '\n\r'):raise ValueError('Literal marker required')
        if type(r.get('reverse')) is not bool or not 0<=r.get('confidence',0)<=.8 or not -5<=r.get('priority',0)<=5:raise ValueError('Invalid rule bounds')
    return copy.deepcopy(rules)


def interpret(sentence,rules,context='global',world='',source='direct text input'):
    from structured_observations import clean,parse_definition
    text=clean(sentence);frames=[]
    negated=any(w in text.split() for w in ('не','нет'))
    modal=any(w in text.split() for w in ('возможно','вероятно','может'))
    for r in validate_rules(rules):
        marker=' '+r['marker'].casefold().strip()+' '
        if r.get('context') and r['context']!=context:continue
        if text.count(marker)!=1:continue
        a,b=text.split(marker)
        if not a or not b or max(len(a.split()),len(b.split()))>8:continue
        # No unsafe collapse of a compound clause into an atomic assertion.
        if any(c in a+b for c in ',;:') or any(w in (a+' '+b).split() for w in ('и','или','если','который')):continue
        if r['reverse']:a,b=b,a
        frame=SemanticFrame(sentence,a,b,r['marker'],r['relation'],'reverse' if r['reverse'] else 'forward',negated,
            'uncertain' if modal else 'asserted',context=context,world=world,source=source,
            confidence=r['confidence'],status='hypothesis' if negated or modal else 'observed',rule_id=r['id'])
        frame.id=hashlib.sha256((sentence+'\0'+context+'\0'+source+'\0'+r['id']).encode()).hexdigest()[:24]
        frames.append((r.get('priority',0),frame))
    if not frames:
        p=parse_definition(sentence)
        if p.get('ok'):
            f=SemanticFrame(sentence,p['subject'],p['target'],'nominal definition',p['relation'],context=context,world=world,source=source,rule_id='legacy.nominal')
            f.id=hashlib.sha256((sentence+context+source).encode()).hexdigest()[:24];return [asdict(f)]
        return []
    best=max(p for p,f in frames);selected=[asdict(f) for p,f in frames if p==best]
    alternatives=[dict(subject=f['subject'],object=f['object'],relation=f['relation'],rule_id=f['rule_id']) for f in selected]
    if len({(f['subject'],f['relation'],f['object']) for f in selected})>1:
        for f in selected:f['status']='hypothesis';f['alternatives']=alternatives
    return selected


def ingest(membrane,sentence):
    from structured_observations import store,parse_definition,resolve_label
    from engine import CrystalReason
    g=membrane.lattice;s=state(g)
    seen=s.setdefault('seen_sentences',[])
    if [sentence,g.active_context] not in seen:seen.append([sentence,g.active_context])
    s['seen_sentences']=seen[-1000:]
    frames=interpret(sentence,s['rules'],g.active_context,g.semantic_world or '')
    for f in frames:
        if not any(old['id']==f['id'] for old in s['frames']):s['frames'].append(f)
        if f['status']!='observed' or f['negated'] or f['alternatives']:continue
        if f['rule_id']=='legacy.nominal':store(membrane,dict(parse_definition(sentence),sentence=sentence));continue
        a=resolve_label(membrane,f['subject'],True);b=resolve_label(membrane,f['object'],True)
        if not a or not b:continue
        if f['relation'] in ('CAUSE','EFFECT'):continue  # Preserve the existing causal-hypothesis gate.
        kind=REGISTRY[f['relation']]['code'];ra,rb=g.label_to_id[a],g.label_to_id[b]
        old=g.edge_meta.get((ra,rb,kind))
        if old and old.metadata.get('status') in ('confirmed','user_observed'):continue
        reason=CrystalReason('input',source_label=f['source'],confidence=f['confidence'],context=f['context'],metadata={'status':'observed','sentence':sentence,'frame_id':f['id'],'rule_id':f['rule_id']})
        g.connect_ids(ra,rb,50,kind,reason);g.relation_observations.append(reason)
    s['frames']=s['frames'][-1000:]
    if not frames:store(membrane,dict(parse_definition(sentence),sentence=sentence))
    return frames


def questions(g):
    s=state(g);result=[];seen=set()
    for f in s['frames']:
        key=(f['sentence'],f['context'])
        if f['id'] in s['questions'] or key in seen:continue
        seen.add(key)
        if not f['alternatives'] and f['confidence']>=.7 and f['status']!='hypothesis':continue
        result.append(dict(id=f['id'],sentence=f['sentence'],candidates=f['alternatives'] or [f],
            reason='competing_interpretations' if f['alternatives'] else 'uncertain_interpretation',
            question='Какая интерпретация верна?',information_value=len(f['alternatives']) or 1))
    return sorted(result,key=lambda x:(-x['information_value'],x['id']))[:20]


def correct(g,frame_id,response,subject='',relation='',object='',source='local user'):
    s=state(g);f=next((f for f in s['frames'] if f['id']==frame_id),None)
    if f is None:raise ValueError('Unknown observed frame')
    if frame_id in s['questions']:raise ValueError('Question already answered')
    if response not in ('correct','incorrect','unknown','skip'):raise ValueError('Invalid response')
    record=dict(frame_id=frame_id,response=response,source=source)
    s['questions'][frame_id]=response
    # All candidates from one observation are the same question.
    for other in s['frames']:
        if other['sentence']==f['sentence'] and other['context']==f['context']:s['questions'][other['id']]=response
    if response in ('unknown','skip'):s['corrections'].append(record);return dict(accepted=False,reason=response)
    if response=='correct':subject,relation,object=f['subject'],f['relation'],f['object']
    if relation not in REGISTRY or not subject.strip() or not object.strip():raise ValueError('Explicit interpretation required')
    example=dict(sentence=f['sentence'],gold=[subject.casefold().strip(),relation,object.casefold().strip()],source=source,context=f['context'])
    from imitation_learning import calibrate_interpretation
    result=calibrate_interpretation(g,example,f['rule_id'])
    record.update(example=example,result=result);s['corrections'].append(record)
    if not any(c['sentence']==example['sentence'] for c in s['controls']):s['controls'].append(example)
    return result


def rollback(g):
    s=state(g)
    if not s['previous']:return False
    s['rules']=s['previous'].pop();s['version']+=1;return True
