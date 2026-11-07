"""Small, auditable nominal grammar. Outputs observations, never verified facts."""
import re
from engine import CrystalReason,EDGE_IS_A,EDGE_PART_OF


def clean(text):return ' '.join(text.casefold().strip(' .!?«»"').split())


def parse_definition(sentence):
    text=sentence.strip(' .!?');lower=clean(text)
    if not lower:return dict(ok=False,reason='empty')
    if re.search(r'\b(не|если|ли|возможно|вероятно)\b',lower):return dict(ok=False,reason='negation_or_modal_scope',attempted=True)
    for marker,kind in [(' происходит от ','ORIGIN'),(' принадлежит ','OWNERSHIP'),(' растёт на ','ORIGIN_OR_LOCATION'),(' растет на ','ORIGIN_OR_LOCATION')]:
        if marker in lower:return dict(ok=False,reason='unsupported_relation_type',relation=kind,attempted=True,sentence=sentence)
    marker=None;instrumental=False
    if re.search(r'\s+является\s+частью\s+',lower):marker=r'\s+является\s+частью\s+';kind='PART_OF'
    elif re.search(r'\s+является\s+',lower):marker=r'\s+является\s+';kind='IS_A';instrumental=True
    elif re.search(r'\s+(?:[—–-]\s*)?это\s+',lower):marker=r'\s+(?:[—–-]\s*)?это\s+';kind='IS_A'
    elif re.search(r'\s+[—–]\s+',lower):marker=r'\s+[—–]\s+';kind='IS_A'
    if marker is None:return dict(ok=False,reason='no_supported_definition',attempted=False)
    parts=re.split(marker,lower)
    if len(parts)!=2:return dict(ok=False,reason='ambiguous_copula',attempted=True)
    subject,target=parts
    if not subject or not target or re.search(r'\b(это|является|который|которая|где|но|или|либо|и)\b',target):
        return dict(ok=False,reason='complex_or_ambiguous_definition',attempted=True)
    words=target.split()
    if len(subject.split())>4 or len(words)>6:return dict(ok=False,reason='phrase_budget',attempted=True)
    if kind=='IS_A' and (words[-1].endswith(('ый','ий','ая','яя','ое','ее','ым','им','ной','кой','чей','щей','шей')) or words[-1] in ('цвета','размера','вкуса')):
        return dict(ok=False,reason='unsupported_relation_type',relation='PROPERTY',attempted=True)
    canonical=list(words);normalization='surface_identity'
    if instrumental and len(words[0])>4 and words[0].endswith('ом'):
        canonical[0]=words[0][:-2];normalization='regular_instrumental_head_candidate'
    return dict(ok=True,attempted=True,relation=kind,subject=subject,target=' '.join(canonical),
        target_surface=target,normalization=normalization,sentence=sentence,
        qualifier={'surface':' '.join(canonical[1:]),'role':'part_of_class_phrase_not_a_separate_origin_claim'} if len(canonical)>1 else None)


def candidate_labels(membrane,text):
    g=membrane.lattice;key=clean(text)
    labels=[label for label in g.observation_aliases.get(key,[]) if label in g.label_to_id]
    if not labels:
        labels=[r.label for r in g.resonators.values() if clean(r.label)==key or clean(membrane._get_surface_form(r.label))==key]
    labels=sorted(set(labels))
    return labels


def resolve_label(membrane,text,create=False):
    g=membrane.lattice;key=clean(text);labels=candidate_labels(membrane,text)
    if len(labels)==1:return labels[0]
    if len(labels)>1:return None
    if create:
        r=g.get_or_create(key);g.observation_aliases[key]=[r.label];return r.label
    return None


def store(membrane,parsed,status='observed',source='direct text input',source_type='text'):
    g=membrane.lattice
    if not parsed.get('ok'):
        if parsed.get('reason')!='empty':
            g.relation_observations.append(CrystalReason('input',source_label=source,source_type=source_type,confidence=0,
                context=g.active_context,timestamp=g.tick_count,metadata={'status':status,'sentence':parsed.get('sentence',''),
                'parse_status':parsed['reason'],'unsupported_relation':parsed.get('relation')}))
        return None
    subject=resolve_label(membrane,parsed['subject'],True);target=resolve_label(membrane,parsed['target'],True)
    if subject is None or target is None:
        parsed['ok']=False;parsed['reason']='ambiguous_concept'
        return store(membrane,parsed,status,source,source_type)
    for surface,label in ((parsed['subject'],subject),(parsed['target'],target),(parsed['target_surface'],target)):
        values=g.observation_aliases.setdefault(clean(surface),[])
        if label not in values:values.append(label)
    a,b=g.label_to_id[subject],g.label_to_id[target]
    for rid in (a,b):g.resonators[rid].inject_energy(membrane.calibration.concept_inject_energy,g.tick_count,cap=membrane.calibration.energy_cap)
    kind={'IS_A':EDGE_IS_A,'PART_OF':EDGE_PART_OF}[parsed['relation']]
    reason=CrystalReason('input',source_label=source,source_type=source_type,confidence=.6 if status=='user_observed' else .55,
        context=g.active_context,timestamp=g.tick_count,metadata={'status':status,'sentence':parsed['sentence'],
        'relation':parsed['relation'],'subject':subject,'target':target,'structured':parsed,'parse_status':'structured_observation'})
    old=g.edge_meta.get((a,b,kind))
    if not (old and old.kind!='hypothesis' and old.metadata.get('status')=='confirmed'):
        # A new ordinary observation does not downgrade an explicit user witness.
        if not (old and old.metadata.get('status')=='user_observed' and status=='observed'):
            g.connect_ids(a,b,50,kind,reason)
    g.relation_observations.append(reason)
    return (subject,parsed['relation'],target)


def surface_label(lattice,label):
    if not label.startswith('root:'):return label
    choices=[key for key,labels in lattice.observation_aliases.items() if label in labels and ':' not in key]
    return min(choices,key=lambda s:(len(s),s)) if choices else label[5:]


def context_applies(lattice,reason,context=None,allow_source=False):
    """Source reports inherit their semantic scope, never another world's facts."""
    chain=lattice.resolve_context(context)
    if reason.context in chain:return True
    node=lattice.contexts.get(reason.context)
    return bool(allow_source and node and node.context_type=='source_observation' and node.parent in chain)
