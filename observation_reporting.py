"""Bounded inspection of parsed and unparsed textual evidence; no inference."""
from engine import EDGE_NAMES
from structured_observations import clean,resolve_label


def report(membrane,query=None):
    g=membrane.lattice;label=resolve_label(membrane,query) if query else None
    relations=[];unparsed=[]
    for key,reason in g.edge_meta.items():
        if len(key)!=3:continue
        a,b,kind=key
        if a not in g.resonators or b not in g.resonators or g.resonators[a].connections.get_type(b,kind) is None:continue
        if label and g.resonators[a].label!=label:continue
        if query and label is None:continue
        if reason.metadata.get('status') not in ('observed','user_observed','confirmed'):continue
        if len(relations)>=100:continue
        relations.append(dict(subject=g.resonators[a].label,relation=EDGE_NAMES[kind],object=g.resonators[b].label,
            status=reason.metadata['status'],source=reason.metadata.get('source_document') or reason.source_label,
            sentence=reason.metadata.get('sentence'),context=reason.context,confidence=reason.confidence))
    for reason in g.relation_observations:
        status=reason.metadata.get('parse_status')
        if not status or status=='structured_observation':continue
        sentence=reason.metadata.get('sentence','')
        if query and clean(query) not in clean(sentence):continue
        if len(unparsed)<100:unparsed.append(dict(sentence=sentence,reason=status,unsupported_relation=reason.metadata.get('unsupported_relation'),
            source=reason.metadata.get('source_document') or reason.source_label,status=reason.metadata.get('status')))
    hypotheses=[dict(source=a,target=b,context=c,status=h['status'],source_document=h['reason'].metadata.get('source_document'),direction_unknown=h['reason'].metadata.get('direction_unknown',False)) for (a,b,c),h in g.causal_hypotheses.items() if not query or (label is not None and a==g.label_to_id[label])][:100]
    return dict(relations=relations,unparsed=unparsed,hypotheses=hypotheses,limit=100,observations=len(g.relation_observations),
                definitions=sum(r['relation']=='IS_A' for r in relations),confirmed=sum(r['status']=='confirmed' for r in relations))
