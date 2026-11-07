"""Conservative interface adapter, not a second calibration engine."""
import hashlib
from engine import EDGE_NAMES,unpack_edge
from imitation_learning import TrainingExample,normalize,instrumental


def aliases(session):
    result={}
    for r in session.population.active.resonators.values():
        for text in (r.label,session.membrane._get_surface_form(r.label)):
            result.setdefault(normalize(text),set()).add(r.label)
    for text,labels in session.population.active.observation_aliases.items():
        result.setdefault(normalize(text),set()).update(label for label in labels if label in session.population.active.label_to_id)
    return result


def question_query(question,known):
    text=normalize(question)
    prefixes=(('что такое ','IS_A'),('чем является ','IS_A'),('частью чего является ','PART_OF'),('что вызывает ','CAUSE'))
    kind='';subject=text
    raw=question.split(maxsplit=1)
    if len(raw)==2 and raw[0] in ('IS_A','PART_OF','CAUSE'):
        kind=raw[0];subject=normalize(raw[1])
    else:
        for prefix,relation in prefixes:
            if text.startswith(prefix):kind=relation;subject=text[len(prefix):];break
    values=known.get(subject,set())
    if len(values)!=1:return None
    label=next(iter(values))
    return (kind+' '+label).strip()


def parse_reference(reference,known):
    from structured_observations import parse_definition
    parsed=parse_definition(reference)
    matches=set()
    if parsed.get('ok'):
        subjects=known.get(normalize(parsed['subject']),set())
        targets=known.get(normalize(parsed['target']),set())
        if len(subjects)==len(targets)==1:matches.add((next(iter(subjects)),parsed['relation'],next(iter(targets))))
    # Preserve original explicit CAUSE/PART_OF grammar for existing clients.
    text=normalize(reference)
    for marker,kind in ((' входит в состав ','PART_OF'),(' вызывает ','CAUSE'),(' приводит к ','CAUSE')):
        if text.count(marker)!=1:continue
        subject,target=text.split(marker);subjects=known.get(subject,set());targets=known.get(target,set())
        if len(subjects)==len(targets)==1:matches.add((next(iter(subjects)),kind,next(iter(targets))))
    return tuple(sorted(matches))


def eligible_relations(session,context,include_user_observed=False):
    from structured_observations import context_applies
    g=session.population.active
    for r in g.resonators.values():
        for target,packed in r.connections.typed_items():
            kind=unpack_edge(packed)[1];reason=g.edge_meta.get((r.id,target,kind))
            if (target in g.resonators and EDGE_NAMES[kind] in ('IS_A','PART_OF','CAUSE') and reason and
                reason.kind!='hypothesis' and
                ((reason.metadata.get('status')=='confirmed' and reason.confidence>=.8) or
                 (include_user_observed and reason.metadata.get('status')=='user_observed' and reason.confidence>=.5)) and
                (reason.context in ('global',context) or (include_user_observed and reason.metadata.get('status')=='user_observed' and context_applies(g,reason,context,True)))):
                yield (r.label,EDGE_NAMES[kind],g.resonators[target].label),reason,unpack_edge(packed)[0]


def confirmed(session,context):
    return eligible_relations(session,context,False)


def prepare(session,question,reference,examples=()):
    g=session.population.active;known=aliases(session);query=question_query(question,known)
    relations=parse_reference(reference,known)
    from structured_observations import parse_definition
    parsed=parse_definition(reference)
    context=g.active_context
    # A geometry-view world is not an implicit semantic constraint on a text request.
    world=''
    consistent=bool(query and len(relations)==1)
    if consistent:
        parts=query.split(maxsplit=1)
        requested=parts[0] if len(parts)==2 and parts[0] in ('IS_A','PART_OF','CAUSE') else ''
        subject=parts[1] if requested else query
        consistent=relations[0][0]==subject and (not requested or relations[0][1]==requested)
        if len(g.semantic_bindings.get(subject,[]))>1:consistent=False;parsed['reason']='ambiguous_sense'
    witness=bool(len(relations)==1 and any(r==relations[0] and why.metadata.get('status')=='user_observed' for r,why,_ in eligible_relations(session,context,True)))
    evidence_mode='user_observed' if witness else 'verified'
    example=TrainingExample('simple-'+hashlib.sha256((question+'\0'+reference+'\0'+context).encode()).hexdigest()[:20],
        query or question,(reference,),relations if consistent else (),context=context,world=world,
        correction_source='interactive demonstration; not a knowledge assertion',correction_confidence=1.0,evidence_mode=evidence_mode)
    analysis=dict(original_question=question,normalized_query=query,parsed_relations=relations,
                  reliable=consistent,world=world,knowledge_asserted=False,evidence_mode=evidence_mode,
                  reference_structure=parsed,missing_knowledge=not any(r in relations for r,_,_ in eligible_relations(session,context,True)))
    if not analysis['reliable']:
        analysis['limitation']='Структура: '+parsed.get('reason','неизвестные или неоднозначные понятия / несовпадение вопроса и эталона')+'. Можно явно добавить структурированное свидетельство, если разбор поддержан; иначе нужна экспертная разметка.'
    # Only metadata of test inputs is inspected to exclude them; never test answers.
    excluded={query}
    for e in examples:
        if e.split=='test':excluded.add(question_query(e.query,known) or e.query)
    subjects={r[0] for r in example.relations}
    for q in excluded:
        if q:subjects.add(q.split(maxsplit=1)[-1] if q.split()[0] in ('IS_A','PART_OF','CAUSE') else q)
    controls=[];groups={}
    for relation,reason,_ in sorted(eligible_relations(session,context,witness),key=lambda x:x[0]):
        subject,kind,target=relation
        if world and any(len(g.semantic_bindings.get(label,[]))!=1 or world not in g.semantic_space.senses[g.semantic_bindings[label][0]].worlds for label in (subject,target)):
            continue
        if subject in subjects:continue
        groups.setdefault((subject,kind),[]).append((relation,reason))
    for (subject,kind),group in sorted(groups.items()):
        q=kind+' '+subject
        from structured_observations import surface_label
        refs=tuple(form for r,_ in group for form in (
            (f'{surface_label(g,subject).capitalize()} — это {surface_label(g,r[2])}.',
             f'{surface_label(g,subject).capitalize()} является {instrumental(surface_label(g,r[2]))}.')
            if ' ' not in surface_label(g,r[2]) else (f'{surface_label(g,subject).capitalize()} — это {surface_label(g,r[2])}.',))) if kind=='IS_A' else ()
        controls.append(TrainingExample('control-'+hashlib.sha256(q.encode()).hexdigest()[:16],q,refs,tuple(r for r,_ in group),
            context=context,world=world,evidence_mode='user_observed' if any(why.metadata.get('status')=='user_observed' for _,why in group) else 'verified',correction_source=' | '.join(sorted({reason.metadata.get('source_document') or reason.source_label or reason.source_type for _,reason in group}))[:2000],
            correction_confidence=1.0 if any(why.metadata.get('status')=='user_observed' for _,why in group) else min(reason.confidence for _,reason in group),split='validation'))
        if len(controls)>=8:break
    analysis['control_examples']=[dict(id=c.id,query=c.query,relations=c.relations,source=c.correction_source,evidence_mode=c.evidence_mode,split=c.split) for c in controls]
    analysis['controls']=len(controls)
    analysis['control_policy']='Other subjects with confirmed relations or explicitly allowed user_observed demonstrations; consistency controls, not independent truth verification'
    return example,controls,analysis
