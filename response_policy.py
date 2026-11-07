"""Evidence-aware response modes, separate from verification and strategy learning."""
import math
import re
from engine import EDGE_NAMES,unpack_edge
from structured_observations import clean,candidate_labels,surface_label,context_applies

MODES={'auto','verified','attributed','hypothesis','associative','unknown','experimental'}


def subject_query(membrane,query):
    query=query.strip().rstrip('?!.');requested=''
    for prefix,kind in [('что такое ','IS_A'),('чем является ','IS_A'),('частью чего является ','PART_OF'),('что вызывает ','CAUSE')]:
        if query.casefold().startswith(prefix):requested=kind;query=query[len(prefix):];break
    pieces=query.split(maxsplit=1)
    if len(pieces)==2 and pieces[0] in ('IS_A','PART_OF','CAUSE'):requested=pieces[0];query=pieces[1]
    labels=candidate_labels(membrane,query)
    return (labels[0] if len(labels)==1 else None),requested,labels


def respond(membrane,query,mode='auto'):
    if mode not in MODES:raise ValueError('Unknown answer mode')
    if mode=='experimental':
        from imitation_learning import ExperimentalLearningLoop
        return ExperimentalLearningLoop(membrane).attempt(query)
    from svo_observations import answer as svo_answer
    svo=svo_answer(membrane,query,mode)
    if svo is not None:return svo
    g=membrane.lattice;label,requested,possible=subject_query(membrane,query)
    def unknown(code,detail):
        return dict(mode='Unknown',text='Недостаточно подтвержденных оснований. '+detail,status='unknown',confidence=0,
            reason=code,relations=[],sources=[],requested_mode=mode)
    if label is None:return unknown('ambiguous_concept' if possible else 'unknown_concept','Понятие неоднозначно.' if possible else 'Понятие неизвестно.')
    if len(g.semantic_bindings.get(label,[]))>1:return unknown('ambiguous_sense','Для слова есть несколько значений; нужен явный смысловой контекст.')
    rid=g.label_to_id[label];r=g.resonators[rid]
    if mode=='unknown':return unknown('explicit_unknown','Выбран режим Unknown.')
    if mode=='associative':return dict(mode='Associative',text='[Ассоциативный эксперимент] '+membrane._generate_associative([label]),status='experimental',confidence=None,relations=[],sources=[])
    candidates=[];observed=False;conflict=False;outside_context=False
    for target,packed in r.connections.typed_items():
        if target not in g.resonators:continue
        _,kind=unpack_edge(packed);name=EDGE_NAMES[kind]
        if name not in ('IS_A','PART_OF','CAUSE','EFFECT','COND','EXCEPT','ANALOG','GOAL'):continue
        if requested and requested!=name:continue
        reason=g.edge_meta.get((rid,target,kind))
        if reason is None:continue
        if not math.isfinite(reason.confidence) or not 0<=reason.confidence<=1:continue
        status=reason.metadata.get('status','unknown')
        source_report=status in ('observed','user_observed') and reason.kind!='hypothesis'
        if not context_applies(g,reason,allow_source=source_report):outside_context=True;continue
        if status in ('contradiction','conflict') or r.belief_status in ('contradiction','defeated'):conflict=True;continue
        level='Verified' if status=='confirmed' and reason.kind!='hypothesis' else 'Attributed' if status in ('observed','user_observed') and reason.kind!='hypothesis' else 'Hypothesis' if reason.kind=='hypothesis' else None
        observed|=level=='Attributed'
        if level is None:continue
        if mode!='auto' and level.casefold()!=mode:continue
        candidates.append(({'Verified':3,'Attributed':2,'Hypothesis':1}[level],unpack_edge(packed)[0],target,name,reason,level))
    if mode in ('auto','hypothesis'):
        for (a,b,context),h in g.causal_hypotheses.items():
            if a==rid and b in g.resonators and context in g.resolve_context() and h['status']=='hypothesis' and requested in ('','CAUSE'):
                candidates.append((1,0,b,'CAUSE',h['reason'],'Hypothesis'))
    if conflict:return unknown('conflict','Есть конфликт свидетельств.')
    if not candidates:
        if observed:return unknown('unconfirmed_observation','Наблюдение есть, но оно не подтверждено; доступен режим Attributed.')
        failures=[o.metadata.get('parse_status') for o in g.relation_observations if o.metadata.get('parse_status') not in (None,'structured_observation') and
                  re.search(r'(?<!\w)'+re.escape(clean(surface_label(g,label)))+r'(?!\w)',clean(o.metadata.get('sentence','')))]
        if failures:return unknown('unsupported_parse','Разбор ограничен: '+', '.join(sorted(set(failures)))+'. Смысловое отношение не утверждается.')
        if mode=='verified' and any(a==rid and h['status']=='hypothesis' for (a,b,c),h in g.causal_hypotheses.items()):
            return unknown('unconfirmed_hypothesis','Есть неподтверждённая гипотеза; Verified не разрешён.')
        if outside_context:return unknown('context_mismatch','Отношение существует в другом смысловом контексте; оно неприменимо к текущему запросу.')
        return unknown('missing_relation','Подходящего смыслового отношения нет; SYN не является фактом.')
    _,_,target,kind,reason,level=max(candidates,key=lambda x:(x[0],x[4].confidence if x[5]!='Verified' else 0,x[1],-x[2]))
    subject=reason.metadata.get('structured',{}).get('subject') or membrane._get_surface_form(label)
    structured=reason.metadata.get('structured',{})
    object_text=structured.get('target') or membrane._get_surface_form(g.resonators[target].label)
    if kind=='IS_A':
        if 'является' in structured.get('sentence','').casefold() and structured.get('normalization')=='surface_identity':text=structured['sentence'].capitalize()+'.'
        else:text=f'{subject.capitalize()} — это {object_text}.'
    elif kind=='PART_OF':text=f'{subject.capitalize()} — часть {object_text}.'
    elif kind in ('EFFECT','COND','EXCEPT','ANALOG','GOAL'):
        predicate={'EFFECT':'возникает вследствие','COND':'требует условия','EXCEPT':'за исключением','ANALOG':'аналогичен','GOAL':'предназначен для'}[kind]
        text=f'{subject.capitalize()} {predicate} {object_text}.'
    else:
        text=f'{subject.capitalize()} вызывает {object_text}.' if not reason.metadata.get('direction_unknown') else f'Возможная связь «{subject}» и «{object_text}»; направление не установлено.'
    source=reason.metadata.get('source_document') or reason.source_label or reason.source_type
    if level=='Attributed':text=f'По наблюдению «{source}»: '+text
    if level=='Hypothesis':text='Предположение (не подтверждено): '+text
    evidence_status=reason.metadata.get('status','hypothesis')
    status='hypothesis' if level=='Hypothesis' else evidence_status
    return dict(mode=level,text=text,status=status,confidence=min(reason.confidence,.6 if level=='Attributed' else .4 if level=='Hypothesis' else 1),
        reason='supported_relation',evidence_status=evidence_status,relations=[dict(source=rid,target=target,type=kind,subject=label,object=g.resonators[target].label,
        status=status,source_context=reason.context,cross_source_context=reason.context not in g.resolve_context(),direction_unknown=bool(reason.metadata.get('direction_unknown')),sentence=reason.metadata.get('sentence') or reason.metadata.get('text'),context=reason.context)],sources=[source],requested_mode=mode)
