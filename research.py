"""Shared bounded acquisition → observation → thinking → sleep research cycle."""
import json
import copy
import time
import urllib.parse
import urllib.request
from pathlib import Path
from engine import CrystalReason, EDGE_NAMES, unpack_edge
from persistence import CrystalSnapshot

MAX_BYTES=256*1024
MAX_TEXT=12000

class ResearchCancelled(RuntimeError):pass

class WikipediaOnly(urllib.request.HTTPRedirectHandler):
    def __init__(self):self.started=time.monotonic();self.redirects=0
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        self.redirects+=1
        if self.redirects>2 or time.monotonic()-self.started>12:raise ValueError('Wikipedia redirect/time budget exceeded')
        parsed=urllib.parse.urlsplit(newurl)
        if parsed.scheme!='https' or parsed.hostname!='ru.wikipedia.org' or parsed.port not in (None,443) or parsed.username is not None:
            raise ValueError('Unexpected Wikipedia redirect')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def fetch_wikipedia(topic,should_stop=lambda:False):
    url='https://ru.wikipedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',prop='extracts',exintro=1,explaintext=1,titles=topic,format='json'))
    req=urllib.request.Request(url,headers={'User-Agent':'CrystalLab/1.0 (local research; bounded extracts)'})
    # Default proxy and TLS verification are preserved. No user-supplied destination.
    started=time.monotonic();chunks=[];size=0
    with urllib.request.build_opener(WikipediaOnly()).open(req,timeout=8) as response:
        while True:
            if should_stop():raise ResearchCancelled('Cancelled during acquisition')
            if time.monotonic()-started>12:raise TimeoutError('Wikipedia total download budget exceeded')
            chunk=response.read1(min(8192,MAX_BYTES+1-size))
            if not chunk:break
            chunks.append(chunk);size+=len(chunk)
            if size>MAX_BYTES:raise ValueError('Wikipedia response exceeds 256 KiB')
    raw=b''.join(chunks)
    if len(raw)>MAX_BYTES:raise ValueError('Wikipedia response exceeds 256 KiB')
    payload=json.loads(raw.decode('utf8'))
    if not isinstance(payload,dict) or not isinstance(payload.get('query',{}),dict):raise ValueError('Invalid Wikipedia JSON')
    pages=payload.get('query',{}).get('pages',{})
    if not isinstance(pages,dict) or any(not isinstance(p,dict) for p in pages.values()):raise ValueError('Invalid Wikipedia pages')
    page=next((p for key,p in pages.items() if key!='-1' and p.get('extract')),None)
    if page is not None and not isinstance(page.get('extract'),str):raise ValueError('Invalid Wikipedia extract')
    if page is None:raise ValueError('Wikipedia page or introductory extract not found')
    return dict(text=page['extract'][:MAX_TEXT],source=url,title=page.get('title',topic),
                source_type='wiki',truncated=len(page['extract'])>MAX_TEXT,retrieved_at=time.time())


def ingest_observation(membrane,lattice,text,source,source_type='text',question=''):
    from semantic_rules import state
    semantic=state(lattice)
    key=[source,text,lattice.active_context]
    if key in semantic['documents']:
        from membrane import CausalStats
        return CausalStats()
    semantic['documents'].append(key)
    semantic['documents']=semantic['documents'][-1000:]
    frame_start=len(semantic['frames'])
    previous_meta=dict(lattice.edge_meta)
    trusted={key:(copy.deepcopy(reason),lattice.resonators[key[0]].connections.get_type(key[1],key[2]))
             for key,reason in previous_meta.items() if len(key)==3 and key[0] in lattice.resonators and
             reason.kind!='hypothesis' and reason.metadata.get('status')=='confirmed' and
             lattice.resonators[key[0]].connections.get_type(key[1],key[2]) is not None}
    before={(r.id,t,unpack_edge(p)[1]) for r in lattice.resonators.values() for t,p in r.connections.typed_items()}
    previous_observations=len(lattice.relation_observations)
    stats=membrane.inject_text(text)
    for frame in semantic['frames'][frame_start:]:frame['source']=source
    observation_status='user_observed' if source_type=='user_correction' else 'observed'
    for observation in lattice.relation_observations[previous_observations:]:
        if observation.metadata.get('status')!='confirmed':
            observation.metadata['source_document']=source
            if source_type=='user_correction':
                observation.metadata['status']='user_observed'
                if observation.metadata.get('parse_status')=='structured_observation':observation.confidence=.6
    for r in lattice.resonators.values():
        for t,p in r.connections.typed_items():
            key=(r.id,t,unpack_edge(p)[1])
            if key in before:
                current=lattice.edge_meta.get(key)
                if current is not None and current is not previous_meta.get(key) and current.metadata.get('status')!='confirmed':
                    current.metadata['source_document']=source
                    if source_type=='user_correction':current.metadata['status']='user_observed'
                continue
            reason=lattice.edge_meta.get(key)
            if reason is None:
                reason=CrystalReason('input',source_label=source,source_type=source_type,confidence=.4,
                    context=lattice.active_context,timestamp=lattice.tick_count,metadata={'status':'observed'})
                lattice.edge_meta[key]=reason
            if reason.metadata.get('status')!='confirmed':
                reason.metadata['source_document']=source
                if source_type=='user_correction':reason.metadata['status']='user_observed'
    # A new textual observation must not erase an older verified source/weight.
    # Preserve its extracted alternative as evidence, not as a replacement fact.
    for key,(reason,packed) in trusted.items():
        current=lattice.edge_meta.get(key)
        if current is not None and current is not previous_meta[key] and current.metadata.get('status')!='confirmed':
            lattice.relation_observations.append(CrystalReason('input',source_label=source,source_type=source_type,confidence=.4,
                context=lattice.active_context,timestamp=lattice.tick_count,metadata={'status':observation_status,
                'relation':EDGE_NAMES[key[2]],'subject':lattice.resonators[key[0]].label,
                'target':lattice.resonators[key[1]].label,'text':text}))
        lattice.edge_meta[key]=reason
        lattice.resonators[key[0]].connections[key[1]]=packed
        pair=key[:2]
        if pair in previous_meta:lattice.edge_meta[pair]=previous_meta[pair]
    for hypothesis in lattice.causal_hypotheses.values():
        reason=hypothesis['reason']
        if reason.context==lattice.active_context:reason.metadata.setdefault('source_document',source)
    lattice.relation_observations.append(CrystalReason('input',source_label=source,source_type=source_type,
        confidence=.4,context=lattice.active_context,timestamp=lattice.tick_count,
        metadata={'status':observation_status,'text':text,'question':question}))
    return stats


def research_steps(membrane,lattice,topic,ticks=15,*,fetcher=fetch_wikipedia,should_stop=lambda:False,snapshot_path=None):
    if not isinstance(topic,str) or not 0<len(topic.strip())<=200:raise ValueError('Topic limit: 200 characters')
    if not isinstance(ticks,int) or not 0<=ticks<=15:raise ValueError('Thinking limit: 15 ticks')
    def checkpoint():
        if should_stop():raise ResearchCancelled('Research cancelled; original memory preserved')
        edges=sum(len(list(r.connections.typed_items())) for r in lattice.resonators.values())
        if len(lattice.resonators)>500 or edges>4000:raise ValueError('Research budget: 500 nodes / 4000 relations')
    checkpoint();yield dict(stage='fetching',topic=topic,source=getattr(fetcher,'source','https://ru.wikipedia.org'))
    material=fetch_wikipedia(topic,should_stop) if fetcher is fetch_wikipedia else fetcher(topic)
    checkpoint()
    if not isinstance(material.get('text'),str) or not material['text'].strip():raise ValueError('Empty study material')
    if len(material['text'])>MAX_TEXT:raise ValueError('Material exceeds 12000 characters')
    source=material['source'];source_type=material.get('source_type','text')
    yield dict(stage='material',material=material,epistemic_status='external_information_not_proof')
    semantic_context=lattice.active_context
    previous=lattice.contexts.get(semantic_context)
    if previous and previous.context_type=='source_observation':semantic_context=previous.parent or 'global'
    source_epoch=f'{source_type}:{topic}'
    if semantic_context!='global':source_epoch+=' @ '+semantic_context
    lattice.set_epoch(source_epoch)
    source_context=lattice.contexts[lattice.active_context]
    source_context.context_type='source_observation';source_context.parent=semantic_context
    sentences=list(membrane.iter_sentences(material['text']))
    if len(sentences)>50:raise ValueError('Research budget: at most 50 sentences')
    for index,sentence in enumerate(sentences):
        checkpoint();stats=ingest_observation(membrane,lattice,sentence,source,source_type)
        checkpoint();yield dict(stage='observation',sentence=index+1,total_sentences=len(sentences),text=sentence,stats=stats.__dict__,epistemic_status='observed')
    for index in range(ticks):
        checkpoint();lattice.tick();checkpoint();yield dict(stage='tick',iteration=index+1,tick=lattice.tick_count)
    checkpoint();merged,pruned,dreams=lattice.defragment();annealed=lattice.anneal_paradoxes();checkpoint()
    yield dict(stage='sleep',merged=merged,pruned=pruned,dreams=dreams[:30],annealed=annealed)
    if snapshot_path:
        checkpoint();CrystalSnapshot.save(lattice,str(snapshot_path))
        yield dict(stage='snapshot',name=Path(snapshot_path).name)
    checkpoint();yield dict(stage='research_complete',extraction_report=membrane.observation_report(),tick=lattice.tick_count,hypotheses=len(lattice.hypotheses),
        observations=len(lattice.relation_observations),causal_hypotheses=len(lattice.causal_hypotheses),confirmed_relations=sum(
        r.metadata.get('status')=='confirmed' for key,r in lattice.edge_meta.items() if len(key)==3))
