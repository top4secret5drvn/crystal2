"""Literal finite-verb SVO grammar and slot queries over existing SemanticFrames.

Predicate vocabulary is linguistic data, never a subject/object answer table.
No SVO action is converted to an experimentally verified CAUSE edge.
"""
import hashlib
import json
from pathlib import Path

GRAMMAR=json.loads(Path(__file__).with_name('svo_grammar.json').read_text(encoding='utf8'))

def normalized(text):
    return ' '.join(text.casefold().replace('ё','е').strip(' .?!').split())

def lemma(text,kind='noun'):
    text=normalized(text)
    vocabulary=GRAMMAR['predicates' if kind=='verb' else 'noun_forms']
    candidates={base for base,forms in vocabulary.items() if text in {normalized(x) for x in [base]+forms}}
    return next(iter(candidates)) if len(candidates)==1 else text

def parse(sentence,context='global',world='',source='direct text input'):
    from semantic_rules import SemanticFrame
    from dataclasses import asdict
    text=normalized(sentence);words=text.split()
    indexes=[i for i,w in enumerate(words) if any(w in {normalized(x) for x in forms} for forms in GRAMMAR['predicates'].values())]
    if not indexes:return [],'unsupported_predicate_or_grammar'
    if len(indexes)!=1:return [],'multiple_predicates'
    i=indexes[0];left=words[:i];right=words[i+1:]
    negated=bool(left and left[-1]=='не')
    if negated:left=left[:-1]
    modal=bool(left and left[0] in ('возможно','вероятно'))
    if modal:left=left[1:]
    if not 1<=len(left)<=GRAMMAR['max_argument_tokens'] or not 1<=len(right)<=GRAMMAR['max_argument_tokens']:return [],'argument_budget_or_missing_argument'
    if any(c in text for c in ',:;') or any(w in left+right for w in ('и','или','если','не','кто','что','ли','который')):return [],'unsupported_scope_or_question'
    subject=' '.join(left);obj=' '.join(right)
    f=asdict(SemanticFrame(sentence,lemma(subject),lemma(obj),lemma(words[i],'verb'),'SVO',negated=negated,
        modality='uncertain' if modal else 'asserted',context=context,world=world,source=source,
        confidence=.6,status='hypothesis' if modal else 'observed',rule_id='svo.finite_literal.v1'))
    f.update(subject_surface=subject,object_surface=obj,predicate_surface=words[i],normalized_subject=lemma(subject),
        normalized_object=lemma(obj),normalized_predicate=lemma(words[i],'verb'))
    f['id']=hashlib.sha256(json.dumps([sentence,context,source,f['rule_id']],ensure_ascii=False).encode()).hexdigest()[:24]
    f['provenance_id']=f['id']
    return [f],None


def answer(membrane,query,mode):
    from engine import CrystalReason
    from structured_observations import context_applies
    g=membrane.lattice;raw=normalized(query);words=raw.split()
    verbs=[i for i,w in enumerate(words) if lemma(w,'verb') in GRAMMAR['predicates']]
    if not verbs:return None
    def unknown(reason):
        return dict(mode='Unknown',text='Неизвестно: '+reason+'.',reason=reason,status='unknown',confidence=0,relations=[],sources=[],requested_mode=mode)
    if len(verbs)!=1:return unknown('ambiguous_question')
    i=verbs[0];predicate=lemma(words[i],'verb');slot=None;polarity=False
    if i==1 and words[0] in ('кто','что'):
        anchor=lemma(' '.join(words[2:]));slot='subject' if words[0]=='кто' else 'either'
    else:
        fs,error=parse(query)
        if not fs:return unknown(error)
        q=fs[0];slot='check';polarity=q['negated']
    frames=[]
    for f in (getattr(g,'semantic_rules_state',None) or {}).get('frames',[]):
        if f.get('relation')!='SVO' or f['predicate']!=predicate:continue
        reason=CrystalReason('input',context=f['context'])
        if not context_applies(g,reason,allow_source=True):continue
        if f.get('modality')!='asserted' or f.get('alternatives'):continue
        for label in (f['subject'],f['object']):
            if len(g.semantic_bindings.get(label,[]))>1:return unknown('ambiguous_sense')
        if slot=='check':
            if (f['subject'],f['object'])==(q['subject'],q['object']):frames.append((f,'check'))
        else:
            if anchor==f['object']:frames.append((f,'subject'))
            if slot=='either' and anchor==f['subject']:frames.append((f,'object'))
    if not frames:return unknown('missing_relation')
    polarities={}
    for f,s in frames:polarities.setdefault((f['subject'],f['predicate'],f['object']),set()).add(f['negated'])
    if any(len(p)>1 for p in polarities.values()):return unknown('conflict')
    positive=[(f,s) for f,s in frames if f['negated']==polarity]
    if not positive:return unknown('negative_observation_only')
    if mode not in ('auto','attributed'):return unknown('unconfirmed_observation' if mode=='verified' else 'mode_not_applicable')
    values={f[s] if s!='check' else 'yes' for f,s in positive}
    if len(values)!=1 or len({s for f,s in positive})>1:return unknown('ambiguous_answer')
    f,s=positive[0]
    result='Да, согласно наблюдению' if s=='check' else f.get(s+'_surface',f[s])
    return dict(mode='Attributed',text=f'По наблюдению «{f["source"]}»: {result}. Это наблюдение не прошло независимую проверку.',
        answer=result,status=f['status'],confidence=min(.6,f['confidence']),reason='observed_predicate',
        relations=[dict(f,answer_slot=s)],sources=list(dict.fromkeys(x['source'] for x,_ in positive)),requested_mode=mode)
