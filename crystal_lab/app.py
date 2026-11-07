import copy
from dataclasses import asdict
from collections import deque
from contextlib import contextmanager
import heapq
import io
import contextlib
import json
from pathlib import Path
import re
import random
import secrets
import shutil
import threading
import time

from flask import Flask, jsonify, render_template, request, Response
from werkzeug.exceptions import HTTPException
from command_session import CommandSession
from engine import CrystalLattice, CrystalPopulation, EDGE_NAMES, unpack_edge
from membrane import LanguageMembrane
from persistence import CrystalSnapshot
from imitation_learning import TrainingExample, ImitationLearner, Config, answer

SAFE_COMMANDS = {'ввод','ложь','отрицание','думай','сон','отжиг','состояние','спроси',
                 'ответь','объясни','аналогии','связи','сравни','гипотезы','подтверди',
                 'опровергни','контексты','качество','популяция','фреймы'}


def reason_json(reason):
    if reason is None:return None
    # Bounded provenance summary: cyclic parent graphs must not recurse forever.
    encoded=json.dumps(reason.metadata,ensure_ascii=False,default=str)
    metadata=reason.metadata if len(encoded)<=4096 else {'truncated':True,'preview':encoded[:4096]}
    return dict(kind=reason.kind,source=reason.source_label[:2000],source_type=reason.source_type[:200],
                confidence=reason.confidence,context=reason.context,timestamp=reason.timestamp,
                metadata=metadata,parents=[str(r)[:1000] for r in reason.parents[:20]])


def state_map(g):
    return {'nodes':{r.id:(r.label,r.energy,r.belief_status,r.state.name) for r in g.resonators.values()},
            'edges':{(r.id,t,unpack_edge(p)[1]):unpack_edge(p)[0] for r in g.resonators.values() for t,p in r.connections.typed_items()},
            'observation_count':len(g.relation_observations),'hypotheses':set(g.hypotheses),'causal':{k:v['status'] for k,v in g.causal_hypotheses.items()},'parameters':copy.deepcopy((g.imitation_state or {}).get('parameters',{}))}


def changes(before,after):
    added=after['nodes'].keys()-before['nodes'].keys();removed=before['nodes'].keys()-after['nodes'].keys()
    edges=after['edges'].keys()-before['edges'].keys()
    energy=[{'id':i,'before':before['nodes'][i][1],'after':after['nodes'][i][1]} for i in before['nodes'].keys() & after['nodes'].keys() if before['nodes'][i][1]!=after['nodes'][i][1]]
    return dict(added_nodes=[{'id':i,'label':after['nodes'][i][0]} for i in sorted(added)[:100]],
        removed_nodes=sorted(removed)[:100],new_relations=[{'source':s,'target':t,'type':EDGE_NAMES[k]} for s,t,k in sorted(edges)[:100]],
        removed_relations=len(before['edges'].keys()-after['edges'].keys()),
        changed_weights=sum(before['edges'][k]!=after['edges'][k] for k in before['edges'].keys() & after['edges'].keys()),
        beliefs=[dict(id=i,before=before['nodes'][i][2:],after=after['nodes'][i][2:]) for i in sorted(before['nodes'].keys() & after['nodes'].keys()) if before['nodes'][i][2:]!=after['nodes'][i][2:]][:100],
        energy=energy[:100],new_hypotheses=sorted(after['hypotheses']-before['hypotheses'])[:100],
        removed_hypotheses=sorted(before['hypotheses']-after['hypotheses'])[:100],
        new_causal_hypotheses=[dict(source=k[0],target=k[1],context=k[2],status=after['causal'][k]) for k in sorted(after['causal'].keys()-before['causal'].keys())[:100]],
        causal_updates=[dict(source=k[0],target=k[1],context=k[2],before=before['causal'][k],after=after['causal'][k]) for k in sorted(before['causal'].keys() & after['causal'].keys()) if before['causal'][k]!=after['causal'][k]][:100],
        parameters={'before':before['parameters'],'after':after['parameters']},
        totals={'added_nodes':len(added),'removed_nodes':len(removed),'new_relations':len(edges),'energy_changes':len(energy)},
        observations_added=after['observation_count']-before['observation_count'],detail_limit=100)


class Lab:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.backups=self.root/'backups';self.backups.mkdir(exist_ok=True)
        if self.backups.is_symlink():raise ValueError('Backup directory must not be a symlink')
        rules=self.root/'learned_language_rules.json'
        if rules.is_symlink():raise ValueError('Rules file must not be a symlink')
        self.population=CrystalPopulation();self.population.spawn_initial()
        self.session=CommandSession(self.population,LanguageMembrane(self.population.active,data_dir=str(self.root)))
        self.lock=threading.RLock();self.token=secrets.token_urlsafe(32)
        self.busy_job=None
        self.history=deque(maxlen=100);self.examples=[];self.sequence=0

    def path(self,name):
        if not isinstance(name,str) or not re.fullmatch(r'[\w .-]{1,100}\.cry',name) or name.startswith('.'):
            raise ValueError('Use a plain .cry filename inside the workspace')
        stem=name.split('.')[0].strip().upper()
        if stem in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
            raise ValueError('Reserved Windows filename')
        path=self.root/name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):raise ValueError('Path escapes workspace')
        return path

    def crystal(self):
        index=int(request.args.get('instance',self.population.active_index))
        if not 0<=index<len(self.population.crystals):raise ValueError('Unknown instance')
        return self.population.crystals[index]

    def backup(self):
        retained=[p for p in self.backups.iterdir() if p.is_file() and not p.is_symlink()]
        if len(retained)>=200 or sum(p.stat().st_size for p in retained)>512*1024*1024:
            raise ValueError('Backup budget reached; archive backups before continuing')
        self.sequence+=1
        filename=self.backups/f'{time.time_ns()}-{self.sequence}.cry'
        CrystalSnapshot.save(self.population.active,str(filename))
        return filename.name

    @contextmanager
    def mutation(self,action):
        if not self.lock.acquire(blocking=False):raise RuntimeError('Another operation is running')
        try:
            if self.busy_job:raise RuntimeError('Interactive job is running; stop it before changing memory')
            data=request.get_json(silent=True) or {}
            if action not in ('select_instance','load_snapshot_copy') and '_instance' in data and int(data['_instance'])!=self.population.active_index:
                raise ValueError('Active instance changed; refresh before modifying memory')
            g=self.population.active
            if len(g.resonators)>10000 and action not in ('select_instance','load_snapshot_copy','save_snapshot'):raise ValueError('Mutation limit is 10000 nodes; inspection remains available')
            before_count=len(self.population.crystals);rng_state=random.getstate()
            before_index=self.population.active_index;transfer_sequence=self.population._transfer_sequence
            before=state_map(g);old=copy.deepcopy(self.session);old_examples=self.examples
            rules=self.root/'learned_language_rules.json'
            if rules.is_symlink():raise ValueError('Rules file must not be a symlink')
            previous_rules=rules.read_bytes() if rules.exists() else None
            output=io.StringIO()
            with contextlib.redirect_stdout(output):backup=self.backup()
            try:
                yield
                if len(self.population.crystals)>8:raise ValueError('Population budget exceeded')
                if action not in ('select_instance','load_snapshot_copy','save_snapshot') and len(self.population.active.resonators)>10000:
                    raise ValueError('Node budget exceeded; operation rolled back')
            except Exception:
                random.setstate(rng_state)
                self.session=old;self.population=old.population;self.examples=old_examples
                if previous_rules is None:rules.unlink(missing_ok=True)
                else:rules.write_bytes(previous_rules)
                self.history.append(dict(action=action,error=True,backup=backup,time=time.time()))
                raise
            after=state_map(self.population.active)
            self.history.append(dict(action=action,time=time.time(),instance=self.population.active_index,
                backup=backup,diff=changes(before,after) if before_index==self.population.active_index else None,
                instance_switch=before_index!=self.population.active_index,population_count=len(self.population.crystals),
                new_instances=list(range(before_count,len(self.population.crystals))),
                transfer_events=[e for e in self.population.transfer_events if e['sequence']>transfer_sequence],
                diagnostic_log=self.population.active.interference_log[-10:]))
        finally:self.lock.release()


def create_app(workspace):
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=128*1024
    lab=Lab(workspace);app.extensions['lab']=lab
    from .jobs import Jobs
    jobs=Jobs(lab);app.extensions['jobs']=jobs

    @app.before_request
    def local_only():
        host=request.host.split(':')[0].lower()
        if host not in ('localhost','127.0.0.1') or request.remote_addr not in ('127.0.0.1','::1'):
            return jsonify(error='Local requests only'),403
        if request.method!='GET':
            if request.headers.get('X-Lab-Token')!=lab.token:return jsonify(error='Invalid local session token'),403
            origin=request.headers.get('Origin')
            if origin and origin!=request.host_url.rstrip('/'):return jsonify(error='Cross-origin mutation refused'),403
            if not request.is_json:return jsonify(error='JSON required'),415

    @app.after_request
    def headers(response):
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options']='nosniff';response.headers['Cache-Control']='no-store'
        return response

    @app.errorhandler(Exception)
    def errors(error):
        status=error.code if isinstance(error,HTTPException) else (409 if isinstance(error,RuntimeError) else 400 if isinstance(error,(ValueError,KeyError,TypeError)) else 500)
        app.logger.info('Lab request failed: %s',error)
        return jsonify(error=str(error) if status!=500 else 'Operation failed; inspect local server log'),status

    def body():
        data=request.get_json()
        if not isinstance(data,dict):raise ValueError('JSON object required')
        return data

    def node(g,r):
        return dict(id=str(r.id),label=r.label,energy=r.energy,state=r.state.name,status=r.belief_status,
                    confidence=r.belief_confidence,context=r.belief_context,reason=reason_json(r.belief_reason),
                    service=r.label=='SELF' or r.label.startswith(('mdl:','mod:','cluster:','skill:','EPOCH:')),
                    hypothesis=r.id in g.hypotheses)

    @app.get('/')
    def home():return render_template('index.html',token=lab.token)

    @app.get('/api/status')
    def status():
        with lab.lock:
            g=lab.crystal()
            return jsonify(active=lab.population.active_index,nodes=len(g.resonators),
                relations=sum(len(list(r.connections.typed_items())) for r in g.resonators.values()),
                hypotheses=len(g.hypotheses),node_hypotheses=len(g.hypotheses),
                causal_hypotheses=sum(h['status']=='hypothesis' for h in g.causal_hypotheses.values()),
                hypotheses_total=len(g.hypotheses)+sum(h['status']=='hypothesis' for h in g.causal_hypotheses.values()),tick=g.tick_count,context=g.active_context,dormant=g.is_dormant,
                geometry_mode=g.geometry_mode,imitation=bool(g.imitation_state),history_count=len(lab.history),
                commands=sorted(SAFE_COMMANDS),busy_job=lab.busy_job)

    @app.get('/api/instances')
    def instances():
        with lab.lock:return jsonify([dict(id=i,active=i==lab.population.active_index,dormant=g.is_dormant,nodes=len(g.resonators),tick=g.tick_count) for i,g in enumerate(lab.population.crystals)])

    @app.post('/api/instances/select')
    def select():
        data=body();index=int(data['id'])
        with lab.mutation('select_instance'):
            if not 0<=index<len(lab.population.crystals):raise ValueError('Unknown instance')
            lab.population.active_index=index
            lab.session.membrane=LanguageMembrane(lab.population.active,data_dir=str(lab.root))
            lab.session.last_state_cache=None
        return jsonify(ok=True)

    @app.get('/api/graph')
    def graph():
        with lab.lock:
            g=lab.crystal();limit=min(300,max(1,int(request.args.get('limit',150))))
            query=request.args.get('q','')[:200].casefold();hide=request.args.get('services','false')!='true'
            relation=request.args.get('type','');scope=request.args.get('scope','all')
            root=request.args.get('node')
            confirmed_ids=set()
            if scope=='confirmed':
                for (source,target,*kinds),reason in g.edge_meta.items():
                    if kinds and reason.kind!='hypothesis' and reason.metadata.get('status')=='confirmed':
                        confirmed_ids.update((source,target))
            eligible=None
            if root is not None:
                rid=int(root)
                if rid not in g.resonators:raise ValueError('Unknown node')
                eligible={rid};eligible.update(g.resonators[rid].connections.keys())
                for r in g.resonators.values():
                    if rid in r.connections:eligible.add(r.id)
            def candidates():
                for r in g.resonators.values():
                    if eligible is not None and r.id not in eligible:continue
                    data=node(g,r)
                    if hide and data['service'] or query and query not in r.label.casefold():continue
                    if scope=='hypotheses' and not data['hypothesis']:continue
                    if scope=='confirmed' and r.belief_status!='confirmed' and r.id not in confirmed_ids:continue
                    yield r
            picked=heapq.nlargest(limit,candidates(),key=lambda r:(r.energy,-r.id));ids={r.id for r in picked}
            edges=[];truncated=False
            for r in picked:
                for target,packed in r.connections.typed_items():
                    w,kind=unpack_edge(packed);name=EDGE_NAMES[kind]
                    if target not in ids or relation and relation!=name:continue
                    if len(edges)>=1200:truncated=True;break
                    reason=g.edge_meta.get((r.id,target,kind))
                    confirmed=bool(reason and reason.kind!='hypothesis' and reason.metadata.get('status')=='confirmed')
                    if scope=='confirmed' and not confirmed:continue
                    edges.append(dict(id=f'{r.id}:{target}:{kind}',source=str(r.id),target=str(target),type=name,weight=w,
                        confirmed=confirmed,provenance=reason_json(reason)))
            return jsonify(nodes=[node(g,r) for r in picked],edges=edges,limit=limit,edge_limit=1200,
                           partial=len(picked)<len(g.resonators) or truncated,total_nodes=len(g.resonators))

    @app.get('/api/nodes/<int:rid>')
    def details(rid):
        with lab.lock:
            g=lab.crystal()
            if rid not in g.resonators:raise ValueError('Unknown node')
            r=g.resonators[rid];data=node(g,r)
            data['neighbors']=[dict(id=str(t),label=g.resonators[t].label if t in g.resonators else '(missing)',type=EDGE_NAMES[unpack_edge(p)[1]],weight=unpack_edge(p)[0],reason=reason_json(g.edge_meta.get((rid,t,unpack_edge(p)[1])))) for t,p in list(r.connections.typed_items())[:300]]
            data['hypothesis_reason']=reason_json(g.hypotheses.get(rid))
            data['causal_hypotheses']=[dict(source=a,target=b,context=context,status=h['status'],reason=reason_json(h['reason'])) for (a,b,context),h in g.causal_hypotheses.items() if rid in (a,b)][:50]
            return jsonify(data)

    @app.get('/api/relations/<int:source>/<int:target>/<kind>')
    def relation_details(source,target,kind):
        with lab.lock:
            g=lab.crystal();types={v:k for k,v in EDGE_NAMES.items()}
            if kind not in types or source not in g.resonators:raise ValueError('Unknown relation')
            packed=g.resonators[source].connections.get_type(target,types[kind])
            if packed is None:raise ValueError('Unknown relation')
            return jsonify(source=source,target=target,type=kind,weight=unpack_edge(packed)[0],reason=reason_json(g.edge_meta.get((source,target,types[kind]))))

    @app.get('/api/geometry')
    def geometry():
        with lab.lock:
            s=lab.crystal().semantic_space
            if s is None or not s.worlds:return jsonify(worlds=[],senses=[],dimensions=s.dimensions if s else None,untrained=0)
            world=request.args.get('world') or next(iter(s.worlds),None)
            if world not in s.worlds:raise ValueError('Unknown world')
            values=[];untrained=0
            for sense in s.senses.values():
                if world not in sense.worlds:continue
                p=s.position(sense.id,world)
                if p is None:untrained+=1
                if len(values)<500:values.append(dict(id=sense.id,label=s.lexemes[sense.lexeme_id].normalized,meaning=sense.meaning_key,role=sense.role,coordinates=p))
            return jsonify(worlds=list(s.worlds),world=world,senses=values,dimensions=s.dimensions,untrained=untrained,limit=500,version=s.version)

    @app.get('/api/history')
    def history():
        with lab.lock:return jsonify(list(lab.history))

    @app.post('/api/commands')
    def command():
        text=body().get('command','')
        if not isinstance(text,str) or not 0<len(text)<=2000:raise ValueError('Command must contain 1–2000 characters')
        parts=text.strip().split(maxsplit=1)
        if not parts:raise ValueError('Empty command')
        action=parts[0].lower();args=parts[1:]
        if action not in SAFE_COMMANDS:raise ValueError('Command is not allowed in Crystal Lab')
        if action=='думай' and args and (not args[0].isdigit() or not 1<=int(args[0])<=10):raise ValueError('Thinking budget is 1–10 ticks')
        with lab.mutation(text):
            count=len(lab.population.active.resonators)
            edges=sum(len(list(r.connections.typed_items())) for r in lab.population.active.resonators.values())
            if edges>40000:raise ValueError('Command budget: at most 40000 typed edges')
            if action in ('сон','отжиг','сравни','качество','спроси') and (count>500 or edges>4000):
                raise ValueError('Analytical command budget: at most 500 nodes / 4000 relations')
            result=lab.session.execute(text)
            if result.error:raise ValueError(result.output[-8000:])
        return jsonify(output=result.output[-30000:],history=list(lab.history)[-1],answer=lab.session.last_response if action=='ответь' else None)

    @app.get('/api/observations')
    def observations_report():
        query=request.args.get('query')
        if query is not None and len(query)>2000:raise ValueError('Question limit: 2000 characters')
        with lab.lock:
            if int(request.args.get('instance',lab.population.active_index))!=lab.population.active_index:raise ValueError('Observation report uses the active instance')
            return jsonify(lab.session.membrane.observation_report(query))

    @app.get('/api/answer')
    def ordinary_answer():
        query=request.args.get('query','');mode=request.args.get('mode','auto')
        if not 0<len(query)<=2000:raise ValueError('Question limit: 2000 characters')
        if mode=='associative':raise ValueError('Associative mode requires explicit POST')
        with lab.lock:
            if int(request.args.get('instance',lab.population.active_index))!=lab.population.active_index:raise ValueError('Questions use the active instance; activate the selected one first')
            return jsonify(lab.session.membrane.respond(query,mode))

    @app.post('/api/answer')
    def associative_answer():
        data=body();query=data.get('query','')
        if not isinstance(query,str) or not 0<len(query)<=2000:raise ValueError('Question limit: 2000 characters')
        with lab.mutation('associative_answer'):
            g=lab.population.active
            if len(g.resonators)>500 or sum(1 for r in g.resonators.values() for _ in r.connections.typed_items())>4000:raise ValueError('Associative answer budget: 500 nodes / 4000 relations')
            result=lab.session.membrane.respond(query,'associative')
        return jsonify(result)

    @app.get('/api/snapshots')
    def snapshots():
        with lab.lock:return jsonify(sorted(p.name for p in lab.root.glob('*.cry') if not p.is_symlink()))

    @app.post('/api/snapshots/load')
    def load():
        source=lab.path(body()['name'])
        with lab.mutation('load_snapshot_copy'):
            if source.stat().st_size>32*1024*1024:raise ValueError('Snapshot limit: 32 MiB')
            copy_path=lab.backups/f'import-{time.time_ns()}.cry';shutil.copy2(source,copy_path)
            g=CrystalLattice()
            if not CrystalSnapshot.load(g,str(copy_path)):raise ValueError('Snapshot load failed; memory preserved')
            lab.population.crystals.append(g);lab.population.active_index=len(lab.population.crystals)-1
            lab.session.membrane=LanguageMembrane(g,data_dir=str(lab.root));lab.session.last_state_cache=None
        return jsonify(ok=True,instance=lab.population.active_index)

    @app.post('/api/snapshots/save')
    def save():
        data=body();path=lab.path(data['name'])
        with lab.mutation('save_snapshot'):
            if path.exists():
                if path.stat().st_size>32*1024*1024:raise ValueError('Overwrite backup limit: 32 MiB')
                if data.get('confirm_overwrite') is not True:raise ValueError('Explicit overwrite confirmation required')
                shutil.copy2(path,lab.backups/f'overwrite-{time.time_ns()}.cry')
            temp=lab.root/f'.save-{time.time_ns()}.tmp'
            try:CrystalSnapshot.save(lab.population.active,str(temp));temp.replace(path)
            finally:temp.unlink(missing_ok=True)
        return jsonify(ok=True)

    @app.get('/api/training')
    def training_state():
        with lab.lock:return jsonify(examples=[asdict(e) for e in lab.examples],state=lab.population.active.imitation_state)

    @app.post('/api/training/examples')
    def import_examples():
        rows=body()['examples']
        if not isinstance(rows,list) or len(rows)>100:raise ValueError('At most 100 examples')
        examples=[TrainingExample(**e) for e in rows]
        if len({e.id for e in examples})!=len(examples):raise ValueError('Duplicate IDs')
        with lab.mutation('import_training_examples'):lab.examples=examples
        return jsonify(ok=True)

    @app.post('/api/training/run')
    def train():
        data=body()
        with lab.mutation('imitation_calibration'):
            example=next((e for e in lab.examples if e.id==data['id'] and e.split=='train'),None)
            if example is None:raise ValueError('Select a train example')
            if 'reference' in data:
                from dataclasses import replace
                if not isinstance(data['reference'],str) or len(data['reference'])>2000:raise ValueError('Reference limit: 2000 characters')
                example=replace(example,references=(data['reference'],))
            g=lab.population.active
            if len((g.imitation_state or {}).get('history',[]))>=100:raise ValueError('Training history budget reached; archive this experiment')
            learner=ImitationLearner(g,[e for e in lab.examples if e.split=='validation'],g.imitation_state)
            result=learner.train_on_example(example,Config(enabled=True,max_candidates=8,max_evaluations=200,guard_previous=True))
            g.imitation_state=learner.state()
        return jsonify(result)

    @app.post('/api/training/rollback')
    def rollback():
        body()
        with lab.mutation('imitation_rollback'):
            g=lab.population.active
            if not g.imitation_state:raise ValueError('No learned state')
            learner=ImitationLearner(g,state=g.imitation_state)
            if not learner.rollback():raise ValueError('No previous parameters')
            g.imitation_state=learner.state()
        return jsonify(ok=True)

    @app.post('/api/study/start')
    def study_start():
        data=body();topic=data.get('topic')
        if not isinstance(topic,str) or not 0<len(topic.strip())<=200:raise ValueError('Topic limit: 200 characters')
        ticks=data.get('ticks',15)
        if not isinstance(ticks,int) or not 0<=ticks<=15:raise ValueError('Thinking limit: 15 ticks')
        fetcher=app.config.get('RESEARCH_FETCHER')  # server/test injection only; never a user URL
        if 'material' in data and data['material']:
            text=data['material']
            if not isinstance(text,str) or len(text)>12000:raise ValueError('Local material limit: 12000 characters')
            def local_material(topic):return dict(text=text,source='local:'+topic,title=topic,source_type='local_text',truncated=False)
            local_material.source='Локальный материал исследователя'
            fetcher=local_material
        job=jobs.start('study',data,fetcher)
        return jsonify(id=job.id),202

    @app.post('/api/imitation/start')
    def simple_start():
        data=body()
        for key in ('question','reference'):
            if not isinstance(data.get(key),str) or not 0<len(data[key].strip())<=2000:raise ValueError(key+' limit: 1–2000 characters')
        return jsonify(id=jobs.start('imitation',data).id),202

    @app.get('/api/jobs/<job_id>')
    def job_state(job_id):
        if job_id not in jobs.entries:raise ValueError('Unknown job')
        job=jobs.entries[job_id]
        with job.condition:return jsonify(id=job.id,state=job.state,done=job.done,events=job.events)

    @app.get('/api/jobs/<job_id>/events')
    def job_stream(job_id):
        if job_id not in jobs.entries:raise ValueError('Unknown job')
        after=int(request.headers.get('Last-Event-ID','0'))
        if after<0:raise ValueError('Invalid event cursor')
        response=Response(jobs.entries[job_id].stream(after),mimetype='text/event-stream')
        response.headers['X-Accel-Buffering']='no'
        return response

    @app.post('/api/jobs/<job_id>/stop')
    def job_stop(job_id):
        body()
        if job_id not in jobs.entries:raise ValueError('Unknown job')
        job=jobs.entries[job_id]
        with job.condition:
            if job.state=='running':job.cancel.set()
            return jsonify(state=job.state,stop_requested=job.cancel.is_set())

    @app.post('/api/imitation/observation')
    def add_correction_observation():
        data=body()
        for key in ('question','reference'):
            if not isinstance(data.get(key),str) or not 0<len(data[key])<=2000:raise ValueError('Text limit: 2000 characters')
        from research import ingest_observation
        from structured_observations import parse_definition
        from simple_imitation import prepare
        parsed=parse_definition(data['reference'])
        if not parsed.get('ok'):raise ValueError('Структурированное свидетельство не добавлено: '+parsed['reason']+' / '+str(parsed.get('relation','')))
        with lab.mutation('add_unconfirmed_teaching_observation'):
            g=lab.population.active;before=len(g.relation_observations)
            ingest_observation(lab.session.membrane,g,data['reference'],'interactive demonstration','user_correction',data['question'])
            example,controls,analysis=prepare(lab.session,data['question'],data['reference'],lab.examples)
            response=lab.session.membrane.respond(data['question'])
            new_records=g.relation_observations[before:]
            evidence_count=sum(r.metadata.get('parse_status')=='structured_observation' for r in new_records)
        return jsonify(status='user_observed',confirmed=False,structured=parsed,applicable=analysis['reliable'],
            new_evidence=evidence_count,new_observation_records=len(new_records),model_parameters_changed=False,calibration_ready=analysis['reliable'] and not analysis['missing_knowledge'] and bool(controls),
            controls=len(controls),answer=response)

    @app.get('/api/semantic-learning')
    def semantic_learning():
        from semantic_rules import state,questions
        with lab.lock:
            g=lab.population.active
            if int(request.args.get('instance',lab.population.active_index))!=lab.population.active_index:raise ValueError('Semantic learning uses the active instance')
            view=copy.deepcopy(g)
            return jsonify(state=state(view),questions=questions(view))

    @app.post('/api/semantic-learning/correct')
    def semantic_correct():
        from semantic_rules import correct
        data=body()
        if any(not isinstance(data.get(k,''),str) or len(data.get(k,''))>2000 for k in ('id','response','subject','relation','object')):raise ValueError('Correction text budget')
        with lab.mutation('semantic_correction'):
            result=correct(lab.population.active,data['id'],data['response'],data.get('subject',''),data.get('relation',''),data.get('object',''))
        return jsonify(result)

    @app.post('/api/semantic-learning/rollback')
    def semantic_rollback():
        from semantic_rules import rollback
        with lab.mutation('semantic_rollback'):result=rollback(lab.population.active)
        return jsonify(rolled_back=result)

    @app.post('/api/semantic-learning/import')
    def semantic_import():
        from semantic_rules import state,validate_rules
        data=body();rules=validate_rules(data['rules'])
        with lab.mutation('semantic_rule_import'):
            s=state(lab.population.active);s['previous'].append(copy.deepcopy(s['rules']));s['rules']=rules;s['version']+=1
        return jsonify(imported=len(rules))

    return app
