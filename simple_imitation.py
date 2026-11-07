"""Conservative interface adapter, not a second calibration engine."""
import hashlib
from engine import EDGE_NAMES,unpack_edge
from imitation_learning import TrainingExample,normalize,instrumental


def aliases(session):
    result={}
    for r in session.population.active.resonators.values():
        for text in (r.label,session.membrane._get_surface_form(r.label)):
            result.setdefault(normalize(text),set()).add(r.label)
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
    text=normalize(reference)
    patterns=((' является ','IS_A'),(' это ','IS_A'),(' входит в состав ','PART_OF'),(' вызывает ','CAUSE'),(' приводит к ','CAUSE'))
    matches=set()
    for marker,kind in patterns:
        if text.count(marker)!=1:continue
        subject,target=text.split(marker)
        subjects=known.get(subject,set())
        targets=set(known.get(target,set()))
        if kind=='IS_A':
            for alias,labels in known.items():
                if normalize(instrumental(alias))==target:targets.update(labels)
        if len(subjects)==len(targets)==1:matches.add((next(iter(subjects)),kind,next(iter(targets))))
    return tuple(sorted(matches))


def confirmed(session,context):
    g=session.population.active
    for r in g.resonators.values():
        for target,packed in r.connections.typed_items():
            kind=unpack_edge(packed)[1];reason=g.edge_meta.get((r.id,target,kind))
            if (target in g.resonators and EDGE_NAMES[kind] in ('IS_A','PART_OF','CAUSE') and reason and
                reason.kind!='hypothesis' and reason.metadata.get('status')=='confirmed' and
                reason.context in ('global',context) and reason.confidence>=.8):
                yield (r.label,EDGE_NAMES[kind],g.resonators[target].label),reason,unpack_edge(packed)[0]


def prepare(session,question,reference,examples=()):
    g=session.population.active;known=aliases(session);query=question_query(question,known)
    relations=parse_reference(reference,known)
    context=g.active_context
    world=g.semantic_world if g.semantic_space is not None and g.semantic_world in g.semantic_space.worlds else ''
    consistent=bool(query and len(relations)==1)
    if consistent:
        parts=query.split(maxsplit=1)
        requested=parts[0] if len(parts)==2 and parts[0] in ('IS_A','PART_OF','CAUSE') else ''
        subject=parts[1] if requested else query
        consistent=relations[0][0]==subject and (not requested or relations[0][1]==requested)
    example=TrainingExample('simple-'+hashlib.sha256((question+'\0'+reference+'\0'+context).encode()).hexdigest()[:20],
        query or question,(reference,),relations if consistent else (),context=context,world=world,
        correction_source='interactive demonstration; not a knowledge assertion',correction_confidence=1.0)
    analysis=dict(original_question=question,normalized_query=query,parsed_relations=relations,
                  reliable=consistent,world=world,knowledge_asserted=False)
    if not analysis['reliable']:
        analysis['limitation']='Не удалось однозначно разобрать вопрос/эталон. Доступна необязательная экспертная разметка.'
    # Only metadata of test inputs is inspected to exclude them; never test answers.
    excluded={query}
    for e in examples:
        if e.split=='test':excluded.add(question_query(e.query,known) or e.query)
    subjects={r[0] for r in example.relations}
    for q in excluded:
        if q:subjects.add(q.split(maxsplit=1)[-1] if q.split()[0] in ('IS_A','PART_OF','CAUSE') else q)
    controls=[];groups={}
    for relation,reason,_ in sorted(confirmed(session,context),key=lambda x:x[0]):
        subject,kind,target=relation
        if world and any(len(g.semantic_bindings.get(label,[]))!=1 or world not in g.semantic_space.senses[g.semantic_bindings[label][0]].worlds for label in (subject,target)):
            continue
        if subject in subjects:continue
        groups.setdefault((subject,kind),[]).append((relation,reason))
    for (subject,kind),group in sorted(groups.items()):
        q=kind+' '+subject
        surface_subject=session.membrane._get_surface_form(subject)
        refs=tuple(f'{surface_subject.capitalize()} является {instrumental(session.membrane._get_surface_form(r[2]))}.' for r,_ in group) if kind=='IS_A' else ()
        controls.append(TrainingExample('control-'+hashlib.sha256(q.encode()).hexdigest()[:16],q,refs,tuple(r for r,_ in group),
            context=context,world=world,correction_source=' | '.join(sorted({reason.source_label or reason.source_type for _,reason in group}))[:2000],
            correction_confidence=min(reason.confidence for _,reason in group),split='validation'))
        if len(controls)>=8:break
    analysis['controls']=len(controls)
    analysis['control_policy']='Other subjects with already confirmed relations; not independent ground-truth discovery'
    return example,controls,analysis
