"""Bounded asynchronous jobs on private core copies, with replayable SSE events."""
import copy
from dataclasses import asdict
import json
import random
import threading
import time
import uuid

from imitation_learning import ImitationLearner,Config,answer,evaluate,TrainingCancelled
from research import research_steps,ResearchCancelled
from simple_imitation import prepare
from persistence import CrystalSnapshot
from engine import EDGE_NAMES,unpack_edge


def frame(g,focus=()):
    selected=[g.resonators[i] for i in dict.fromkeys(focus) if i in g.resonators][:80]
    selected_ids={r.id for r in selected}
    nodes=selected+sorted((r for r in g.resonators.values() if r.id not in selected_ids),key=lambda r:(-r.energy,r.id))[:80-len(selected)];ids={r.id for r in nodes}
    edges=[]
    for r in nodes:
        for target,packed in r.connections.typed_items():
            if target not in ids or len(edges)>=250:continue
            w,kind=unpack_edge(packed);reason=g.edge_meta.get((r.id,target,kind))
            edges.append(dict(id=f'{r.id}:{target}:{kind}',source=str(r.id),target=str(target),type=EDGE_NAMES[kind],weight=w,
                edge_origin='typed_relation',confirmed=bool(reason and reason.kind!='hypothesis' and reason.metadata.get('status')=='confirmed')))
    causal=[]
    for (a,b,context),hyp in list(g.causal_hypotheses.items())[:50]:
        reason=hyp['reason']
        causal.append(dict(source=a,target=b,context=context,status=hyp['status'],confidence=reason.confidence,
            provenance=dict(kind=reason.kind,source=reason.source_label,source_type=reason.source_type,source_document=reason.metadata.get('source_document'))))
        if a in ids and b in ids and hyp['status']=='hypothesis' and len(edges)<250:
            edges.append(dict(id='hyp:'+str(a)+':'+str(b)+':'+context,source=str(a),target=str(b),type='CAUSE?',weight=0,confirmed=False,edge_origin='causal_hypothesis'))
    node_hypotheses=[dict(id=i,source=r.source_label,source_type=r.source_type,context=r.context,confidence=r.confidence,components=g.abstraction_components.get(i,())) for i,r in list(g.hypotheses.items())[:50]]
    return dict(nodes=[dict(id=str(r.id),label=r.label[:200],energy=r.energy,status=r.belief_status,hypothesis=r.id in g.hypotheses) for r in nodes],edges=edges,
                total_nodes=len(g.resonators),tick=g.tick_count,hypotheses=len(g.hypotheses),node_hypotheses=node_hypotheses,causal_hypotheses=causal,causal_hypothesis_count=len(g.causal_hypotheses),temporary=True)


class Job:
    def __init__(self,kind,index):
        self.id=uuid.uuid4().hex;self.kind=kind;self.instance=index
        self.cancel=threading.Event();self.condition=threading.Condition()
        self.events=[];self.done=False;self.state='running';self.thread=None
    def validate(self,event):
        if len(self.events)>=160:raise ValueError('Job event budget exceeded')
        if len(json.dumps(event,ensure_ascii=False,allow_nan=False).encode('utf8'))>256*1024:raise ValueError('Event payload budget exceeded')
    def emit(self,event):
        with self.condition:
            self.validate(event)
            event=copy.deepcopy(event)
            event.update(sequence=len(self.events)+1,job=self.id,time=time.time())
            self.events.append(event);self.condition.notify_all()
    def stream(self,after=0):
        while True:
            with self.condition:
                pending=self.events[after:]
                done=self.done
                if not pending and not done:self.condition.wait(timeout=10)
            if pending:
                for event in pending:
                    after=event['sequence']
                    yield f"id: {after}\ndata: {json.dumps(event,ensure_ascii=False)}\n\n"
            elif done:return
            else:yield ': keepalive\n\n'


class Jobs:
    def __init__(self,lab):self.lab=lab;self.entries={}
    def start(self,kind,data,fetcher=None):
        lab=self.lab
        with lab.lock:
            if lab.busy_job:raise RuntimeError('Another job is running')
            if '_instance' in data and int(data['_instance'])!=lab.population.active_index:raise ValueError('Active instance changed')
            g=lab.population.active
            if len(g.resonators)>500:raise ValueError('Interactive job limit: 500 nodes')
            if sum(len(list(r.connections.typed_items())) for r in g.resonators.values())>4000:raise ValueError('Interactive job limit: 4000 relations')
            for key in list(self.entries):
                if len(self.entries)<5:break
                if self.entries[key].done:del self.entries[key]
            if len(self.entries)>=5:raise RuntimeError('Previous jobs are still finalizing')
            job=Job(kind,lab.population.active_index);private=copy.deepcopy(lab.session)
            root=lab.root/'jobs'
            if root.is_symlink():raise ValueError('Jobs directory must not be a symlink')
            root.mkdir(exist_ok=True)
            if len(list(root.iterdir()))>=50:raise ValueError('Archive old job directories before continuing')
            stage=root/job.id;stage.mkdir()
            private.membrane.learned_rules_path=str(stage/'learned_language_rules.json')
            backup=lab.backup();before=self._state(g);rng=random.getstate()
            initial=dict(stage='started',kind=kind,instance=job.instance,backup=backup,temporary=True,graph=frame(g))
            try:job.validate(initial)
            except (ValueError,TypeError):stage.rmdir();raise
            lab.busy_job=job.id;self.entries[job.id]=job
            job.emit(initial)
            job.thread=threading.Thread(target=self.run,args=(job,private,data,stage,backup,before,rng,fetcher),daemon=True)
            try:job.thread.start()
            except RuntimeError:
                lab.busy_job=None;del self.entries[job.id];stage.rmdir()
                raise
            return job
    @staticmethod
    def _state(g):
        from .app import state_map
        return state_map(g)
    def run(self,job,session,data,stage,backup,before,rng,fetcher):
        lab=self.lab;g=session.population.active;committed=False
        try:
            if job.kind=='study':
                kwargs=dict(should_stop=job.cancel.is_set,snapshot_path=stage/'research.cry')
                if fetcher:kwargs['fetcher']=fetcher
                for event in research_steps(session.membrane,g,data['topic'],data.get('ticks',15),**kwargs):
                    if job.cancel.is_set():raise ResearchCancelled('Cancelled after current step')
                    event['graph']=frame(g)
                    job.emit(event)
            else:
                example,controls,analysis=prepare(session,data['question'],data['reference'],lab.examples)
                job.emit(dict(stage='analysis',analysis=analysis))
                learner=ImitationLearner(g,controls,g.imitation_state)
                if len(learner.history)>=98:raise ValueError('Archive training history before continuing')
                accepted_any=False
                for round_index in range(2):
                    if job.cancel.is_set():raise TrainingCancelled('Cancelled before next round')
                    def progress(event):
                        event['round']=round_index+1;event['graph']=frame(g,event.get('attempt',{}).get('path',()))
                        job.emit(event)
                    record=learner.train_on_example(example,Config(enabled=True,max_candidates=8,max_evaluations=200,guard_previous=True),
                        on_event=progress,should_stop=job.cancel.is_set)
                    accepted_any=accepted_any or record['accepted']
                    job.emit(dict(stage='round_complete',round=round_index+1,result=record))
                    if not record['accepted']:break
                    result=evaluate(answer(g,example,learner.parameters),example)
                    if result['semantic']==1 and result['form']!=0:break
                if accepted_any:g.imitation_state=learner.state()
            with lab.lock,job.condition:
                if job.cancel.is_set():raise ResearchCancelled('Cancelled before commit')
                if lab.population.active_index!=job.instance:raise ValueError('Active instance changed')
                from .app import changes
                diff=changes(before,self._state(g))
                apply=job.kind=='study' or accepted_any
                if not apply:
                    job.state='completed'
                    lab.history.append(dict(action='imitation_no_change',time=time.time(),instance=job.instance,job=job.id,backup=backup,diff=diff))
                    job.emit(dict(stage='completed',committed=False,diff=diff,graph=frame(lab.population.active),note='No supported improvement; original core state preserved'))
                    return
                snapshot='study-'+job.id+'.cry' if job.kind=='study' else None
                terminal=dict(stage='completed',committed=True,diff=diff,graph=frame(g),snapshot=snapshot,
                    note='Text observations and hypotheses are not confirmed knowledge')
                job.validate(terminal)  # Check serialization/budgets before any publication.
                rules=stage/'learned_language_rules.json';destination=lab.root/'learned_language_rules.json'
                old_rules=stage/'original-rules.json';moved_old=False;moved_new=False;published=False
                target_snapshot=lab.path(snapshot) if snapshot else None
                if destination.is_symlink():raise ValueError('Rules path is a symlink')
                if target_snapshot and target_snapshot.exists():raise ValueError('Snapshot collision')
                try:
                    if rules.exists():
                        if destination.exists():destination.replace(old_rules);moved_old=True
                        rules.replace(destination);moved_new=True
                    if target_snapshot:(stage/'research.cry').replace(target_snapshot);published=True
                except Exception:
                    if published:target_snapshot.unlink(missing_ok=True)
                    if moved_new:destination.unlink(missing_ok=True)
                    if moved_old:old_rules.replace(destination)
                    raise
                lab.session=session;lab.population=session.population;committed=True
                lab.history.append(dict(action=job.kind,time=time.time(),instance=job.instance,job=job.id,backup=backup,diff=diff))
                job.state='completed';job.emit(terminal)
        except (ResearchCancelled,TrainingCancelled) as error:
            job.state='cancelled';job.emit(dict(stage='cancelled',committed=False,message=str(error),graph=frame(lab.population.active)))
        except Exception as error:
            job.state='failed';job.emit(dict(stage='failed',committed=committed,message=str(error),graph=frame(lab.population.active)))
        finally:
            with lab.lock:
                if not committed:random.setstate(rng)
                if not committed:lab.history.append(dict(action=job.kind,time=time.time(),job=job.id,error=True,state=job.state,backup=backup))
            try:(stage/'events.json').write_text(json.dumps(job.events,ensure_ascii=False,indent=2),encoding='utf8')
            except OSError as error:job.emit(dict(stage='report_write_failed',message=str(error)))
            finally:
                with lab.lock:lab.busy_job=None
                with job.condition:job.done=True;job.condition.notify_all()
