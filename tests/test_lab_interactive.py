import copy,json,pickle,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from crystal_lab import create_app
from engine import CrystalReason,EDGE_IS_A,EDGE_PART_OF
from imitation_learning import ImitationLearner,TrainingExample,Config,TrainingCancelled,Parameters
from simple_imitation import prepare
from research import fetch_wikipedia,ingest_observation


class InteractiveLab(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.app=create_app(self.root);self.client=self.app.test_client();self.lab=self.app.extensions['lab']
        self.headers={'X-Lab-Token':self.lab.token}
    def tearDown(self):
        for job in self.app.extensions['jobs'].entries.values():
            job.cancel.set();job.thread.join(10)
        self.directory.cleanup()
    def post(self,path,data):return self.client.post('/api/'+path,json=data,headers=self.headers)
    def finish(self,response):
        self.assertEqual(response.status_code,202,response.json)
        job=self.app.extensions['jobs'].entries[response.json['id']];job.thread.join(10)
        self.assertTrue(job.done);return job
    def knowledge(self):
        g=self.lab.population.active
        for name in ('яблоко','апельсин','абрикос'):
            for target,kind,w in [('фрукт',EDGE_IS_A,70),('предмет',EDGE_IS_A,60),('система',EDGE_PART_OF,90)]:
                g.connect(name,target,w,kind,CrystalReason('observation',source_label='synthetic test',confidence=.9,metadata={'status':'confirmed'}))
        return g

    def test_study_real_steps_sources_and_unconfirmed(self):
        source='https://ru.wikipedia.org/w/api.php?test_fixture'
        self.app.config['RESEARCH_FETCHER']=lambda topic:dict(text='Яблоко это фрукт. Яблоко вызывает рост. Фрукт это пища.',source=source,source_type='wiki',title=topic)
        job=self.finish(self.post('study/start',{'topic':'яблоко','ticks':2}))
        self.assertEqual(job.state,'completed',job.events[-1])
        stages=[e['stage'] for e in job.events]
        self.assertIn('material',stages);self.assertEqual(stages.count('tick'),2);self.assertEqual(stages.count('observation'),3);self.assertIn('sleep',stages)
        g=self.lab.population.active;self.assertGreater(len(g.resonators),1)
        self.assertFalse(any(r.metadata.get('status')=='confirmed' for key,r in g.edge_meta.items() if len(key)==3))
        self.assertTrue(any(r.metadata.get('source_document')==source for key,r in g.edge_meta.items() if len(key)==3))
        self.assertTrue(job.events[-1]['graph']['causal_hypotheses'])
        self.assertTrue(job.events[-1]['diff']['new_causal_hypotheses'])
        self.assertTrue(job.events[-1]['committed']);self.assertTrue(list(self.root.glob('study-*.cry')))

    def test_stream_before_completion_cancel_and_busy(self):
        started=threading.Event();release=threading.Event()
        def provider(topic):
            started.set();release.wait(5)
            return dict(text='Яблоко это фрукт.',source='fixture',source_type='test')
        self.app.config['RESEARCH_FETCHER']=provider
        before=pickle.dumps(self.lab.session)
        response=self.post('study/start',{'topic':'яблоко'});job=self.app.extensions['jobs'].entries[response.json['id']]
        started.wait(5)
        stream=self.client.get('/api/jobs/'+job.id+'/events',buffered=False)
        self.assertIn(b'"stage": "started"',next(iter(stream.response)))
        self.assertFalse(job.done)
        self.assertEqual(self.post('commands',{'command':'думай 1'}).status_code,409)
        self.assertEqual(self.post('jobs/'+job.id+'/stop',{}).status_code,200)
        release.set();job.thread.join(10);stream.close()
        self.assertEqual(job.state,'cancelled');self.assertEqual(before,pickle.dumps(self.lab.session))
        self.assertFalse(list(self.root.glob('study-*.cry')))

    def test_network_failure_preserves_memory(self):
        def fail(topic):raise OSError('network fixture failure')
        self.app.config['RESEARCH_FETCHER']=fail;before=pickle.dumps(self.lab.session)
        job=self.finish(self.post('study/start',{'topic':'яблоко'}))
        self.assertEqual(job.state,'failed');self.assertEqual(before,pickle.dumps(self.lab.session))
        self.assertIn('network fixture failure',job.events[-1]['message'])

    def test_simple_natural_question_real_candidates_transfer_and_rollback(self):
        g=self.knowledge();before=[(r.id,list(r.connections.typed_items()),r.hdc_vector) for r in g.resonators.values()]
        job=self.finish(self.post('imitation/start',{'question':'Что такое яблоко?','reference':'Яблоко является фруктом.'}))
        self.assertEqual(job.state,'completed',job.events[-1])
        self.assertTrue(job.events[-1]['committed'])
        self.assertTrue(any(e['stage']=='candidate' for e in job.events))
        self.assertTrue(any(e['stage']=='initial_attempt' and e['attempt']['text'] for e in job.events))
        learned=self.lab.population.active
        self.assertEqual(before,[(r.id,list(r.connections.typed_items()),r.hdc_vector) for r in learned.resonators.values()])
        self.assertEqual(learned.imitation_state['parameters']['is_a_form'],1)
        self.assertEqual(self.post('training/rollback',{}).status_code,200)

    def test_unknown_semantics_and_missing_knowledge_do_not_commit(self):
        self.knowledge()
        before=pickle.dumps(self.lab.session)
        job=self.finish(self.post('imitation/start',{'question':'Что такое яблоко?','reference':'Яблоко растёт на яблоне.'}))
        self.assertFalse(job.events[-1]['committed']);self.assertEqual(before,pickle.dumps(self.lab.session))
        # Parsed existing endpoints, but PART_OF apple→fruit is not known/confirmed.
        job=self.finish(self.post('imitation/start',{'question':'Частью чего является яблоко?','reference':'Яблоко входит в состав фрукт.'}))
        initial=next(e for e in job.events if e['stage']=='initial_attempt')
        self.assertEqual(initial['diagnosis']['kind'],'D');self.assertFalse(job.events[-1]['committed'])
        self.assertEqual(before,pickle.dumps(self.lab.session))

    def test_controls_exclude_test_queries_and_do_not_read_test_answers(self):
        self.knowledge()
        test=TrainingExample('heldout','IS_A абрикос',('POISONED TEST ANSWER',),split='test')
        e,controls,analysis=prepare(self.lab.session,'Что такое яблоко?','Яблоко является фруктом.',[test])
        self.assertTrue(controls);self.assertTrue(all('абрикос' not in c.query and 'яблоко' not in c.query for c in controls))
        changed=TrainingExample('heldout','IS_A абрикос',('DIFFERENT ANSWER',),split='test')
        self.assertEqual(prepare(self.lab.session,'Что такое яблоко?','Яблоко является фруктом.',[changed]),(e,controls,analysis))
        self.assertTrue(any(len(c.relations)>1 for c in controls if c.query.startswith('IS_A')))

    def test_observation_is_separate_and_not_truth(self):
        response=self.post('imitation/observation',{'question':'Что такое яблоко?','reference':'Яблоко это фрукт.'})
        self.assertEqual(response.status_code,200,response.json);self.assertFalse(response.json['confirmed'])
        self.assertTrue(self.lab.population.active.relation_observations)
        self.assertIsNone(self.lab.population.active.imitation_state)

    def test_learner_callback_cancellation_does_not_apply_candidate(self):
        self.knowledge();example,controls,_=prepare(self.lab.session,'Что такое яблоко?','Яблоко является фруктом.')
        learner=ImitationLearner(self.lab.population.active,controls);stop=threading.Event();events=[]
        def callback(event):
            events.append(event)
            if event['stage']=='candidate':stop.set()
        with self.assertRaises(TrainingCancelled):learner.train_on_example(example,Config(enabled=True),on_event=callback,should_stop=stop.is_set)
        self.assertEqual(learner.parameters,Parameters());self.assertFalse(learner.previous)
        self.assertTrue(any(e['stage']=='candidate' for e in events))

    def test_local_material_and_acquisition_size_limit(self):
        job=self.finish(self.post('study/start',{'topic':'яблоко','material':'Яблоко это фрукт.','ticks':0}))
        material=next(e for e in job.events if e['stage']=='material')['material']
        self.assertEqual(material['source_type'],'local_text')
        self.assertEqual(self.post('study/start',{'topic':'x','material':'x'*12001}).status_code,400)
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read1(self,n):return b'x'*n
        class Opener:
            def open(self,*args,**kwargs):return Response()
        with patch('research.urllib.request.build_opener',return_value=Opener()):
            with self.assertRaisesRegex(ValueError,'256 KiB'):fetch_wikipedia('яблоко')

    def test_question_reference_mismatch_and_no_independent_controls(self):
        self.knowledge()
        e,controls,analysis=prepare(self.lab.session,'Что такое апельсин?','Яблоко является фруктом.')
        self.assertFalse(analysis['reliable']);self.assertFalse(e.relations)
        e,controls,analysis=prepare(self.lab.session,'Что такое яблоко?','Яблоко входит в состав система.')
        self.assertFalse(analysis['reliable'])
        # A single known subject cannot serve as its own independent validation.
        lab=self.app.extensions['lab']
        for label in ('апельсин','абрикос'):
            rid=lab.population.active.label_to_id[label]
            del lab.population.active.resonators[rid];del lab.population.active.label_to_id[label]
        before=pickle.dumps(lab.session)
        job=self.finish(self.post('imitation/start',{'question':'Что такое яблоко?','reference':'Яблоко является фруктом.'}))
        self.assertFalse(job.events[-1]['committed']);self.assertEqual(before,pickle.dumps(lab.session))

    def test_sse_resume_has_no_repeated_ids(self):
        job=self.finish(self.post('study/start',{'topic':'яблоко','material':'Яблоко это фрукт.','ticks':0}))
        response=self.client.get('/api/jobs/'+job.id+'/events',headers={'Last-Event-ID':'3'})
        values=[json.loads(line[6:]) for line in response.data.decode().splitlines() if line.startswith('data: ')]
        self.assertTrue(values);self.assertTrue(all(v['sequence']>3 for v in values))
        self.assertEqual(len(values),len({v['sequence'] for v in values}))

    def test_text_observation_does_not_replace_confirmed_provenance(self):
        g=self.lab.population.active;m=self.lab.session.membrane
        g.connect('яблоко','фрукт',75,EDGE_IS_A,CrystalReason('observation',source_label='verified experiment',metadata={'status':'confirmed'}))
        a,b=g.label_to_id['яблоко'],g.label_to_id['фрукт']
        def parse(text):
            g.connect('яблоко','фрукт',50,EDGE_IS_A,CrystalReason('input',source_type='text',metadata={'status':'observed'}))
        with patch.object(m,'inject_text',side_effect=parse):
            ingest_observation(m,g,'Яблоко это фрукт.','local:test','local_text')
        self.assertEqual(g.edge_meta[a,b,EDGE_IS_A].source_label,'verified experiment')
        self.assertEqual(g.edge_meta[a,b,EDGE_IS_A].metadata['status'],'confirmed')
        self.assertEqual(g.resonators[a].connections.get_type(b,EDGE_IS_A)&0x0fffffff,75)
        self.assertTrue(any(r.source_label=='local:test' and r.metadata.get('status')=='observed' for r in g.relation_observations))

    def test_failed_publication_restores_rules_and_core(self):
        rules=self.root/'learned_language_rules.json';rules.write_text('{}',encoding='utf8')
        before=pickle.dumps(self.lab.session);original_replace=Path.replace
        def fail_snapshot(path,target):
            if path.name=='research.cry':raise OSError('publication failure fixture')
            return original_replace(path,target)
        with patch.object(Path,'replace',fail_snapshot):
            job=self.finish(self.post('study/start',{'topic':'яблоко','material':'Яблоко это фрукт.','ticks':0}))
        self.assertEqual(job.state,'failed');self.assertEqual(before,pickle.dumps(self.lab.session))
        self.assertEqual(rules.read_text(encoding='utf8'),'{}');self.assertFalse(list(self.root.glob('study-*.cry')))
