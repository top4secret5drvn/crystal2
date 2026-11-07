import json,pickle,tempfile,unittest
from pathlib import Path
from engine import CrystalLattice,CrystalReason,EDGE_IS_A,EDGE_SYNTAGM
from membrane import LanguageMembrane
from command_session import CommandSession
from crystal_lab import create_app
from imitation_learning import ImitationLearner,Config
from simple_imitation import prepare
from research import research_steps
from structured_observations import parse_definition

class EvidenceAnswers(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.app=create_app(self.directory.name)
        self.lab=self.app.extensions['lab'];self.client=self.app.test_client();self.headers={'X-Lab-Token':self.lab.token}
    def tearDown(self):
        for j in self.app.extensions['jobs'].entries.values():j.cancel.set();j.thread.join(10)
        self.directory.cleanup()
    def post(self,path,body):return self.client.post('/api/'+path,json=body,headers=self.headers)
    def test_all_definition_forms_and_modes(self):
        m=self.lab.session.membrane
        for sentence,query,target in [('Яблоко — это фрукт.','яблоко','фрукт'),('Яблоко является плодом яблони.','яблоко','плод яблони'),('Апельсин — цитрусовый плод.','апельсин','цитрусовый плод'),('Процессор — это вычислительный модуль.','процессор','вычислительный модуль')]:
            m.inject_text(sentence)
            reply=m.respond(query)
            self.assertEqual(reply['mode'],'Attributed',reply)
            self.assertTrue(any(target in r.label for r in self.lab.population.active.resonators.values()) or target in reply['text'])
            self.assertTrue(reply['sources']);self.assertLessEqual(reply['confidence'],.6)
            self.assertEqual(m.respond(query,'verified')['mode'],'Unknown')
        g=self.lab.population.active
        self.assertTrue(g.relation_observations)
        self.assertFalse(any(r.metadata.get('status')=='confirmed' for k,r in g.edge_meta.items() if len(k)==3))
    def test_syn_origin_property_and_unknown_are_not_facts(self):
        g=self.lab.population.active;m=self.lab.session.membrane
        g.connect('a','b',100,EDGE_SYNTAGM)
        self.assertEqual(m.respond('a')['mode'],'Unknown')
        self.assertEqual(m.respond('missing')['mode'],'Unknown')
        for sentence,kind in [('Яблоко — зелёное.','PROPERTY'),('Вода является холодной.','PROPERTY'),('Яблоко является красным.','PROPERTY'),('Яблоко происходит от яблони.','ORIGIN'),('Яблоко принадлежит садоводу.','OWNERSHIP')]:
            p=parse_definition(sentence);self.assertFalse(p['ok']);self.assertEqual(p['relation'],kind)
            m.inject_text(sentence)
        self.assertFalse(any(r.get_connections_by_type(EDGE_IS_A) for r in g.resonators.values()))
    def test_cli_complete_path(self):
        s=self.lab.session
        for command in ('ввод яблоко это фрукт','спроси яблоко','ответь'):
            r=s.execute(command);self.assertFalse(r.error,r.output)
        self.assertEqual(s.last_response['mode'],'Attributed')
        self.assertIn('фрукт',s.last_response['text'])
    def test_legacy_snapshots_do_not_acquire_verified_status(self):
        from persistence import CrystalSnapshot
        repo=Path(__file__).resolve().parents[1]
        for name in ('bugfix_test.cry','bugfix2.cry'):
            g=CrystalLattice();self.assertTrue(CrystalSnapshot.load(g,str(repo/name)))
            m=LanguageMembrane(g,data_dir=self.directory.name);m.inject_text('Яблоко это фрукт.')
            self.assertEqual(m.respond('яблоко')['mode'],'Attributed')
            self.assertEqual(m.respond('яблоко','verified')['mode'],'Unknown')
    def test_explicit_witness_then_conditional_calibration(self):
        pair={'question':'Что такое яблоко?','reference':'Яблоко — это плод яблони.'}
        m=self.lab.session.membrane;m.inject_text('Яблоко это фрукт.')
        e,c,a=prepare(self.lab.session,pair['question'],pair['reference'])
        self.assertTrue(a['missing_knowledge'])
        # Independent control is explicit user evidence, not a duplicate of the train phrase.
        self.assertEqual(self.post('imitation/observation',{'question':'Что такое апельсин?','reference':'Апельсин — это цитрусовый плод.'}).status_code,200)
        result=self.post('imitation/observation',pair)
        self.assertEqual(result.status_code,200,result.json);self.assertEqual(result.json['status'],'user_observed')
        self.assertFalse(result.json['model_parameters_changed']);self.assertTrue(result.json['calibration_ready'])
        e,c,a=prepare(self.lab.session,pair['question'],pair['reference'])
        self.assertEqual(e.evidence_mode,'user_observed');self.assertTrue(c)
        learner=ImitationLearner(self.lab.population.active,c)
        r=learner.train_on_example(e,Config(enabled=True))
        self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'user_observed_policy_not_enabled')
        r=learner.train_on_example(e,Config(enabled=True,allow_user_observed=True))
        self.assertIn(r['reason'],('no_supported_improvement','train_gain_without_control_regression'))
        self.assertEqual(self.lab.session.membrane.respond('яблоко','verified')['mode'],'Unknown')
        self.assertEqual(self.lab.session.membrane.respond('яблоко')['mode'],'Attributed')
    def test_source_sentence_and_snapshot_aliases(self):
        from persistence import CrystalSnapshot
        m=self.lab.session.membrane;g=self.lab.population.active
        source=lambda q:dict(text='Яблоко является плодом яблони.',source='fixture:article',source_type='wiki')
        list(research_steps(m,g,'яблоко',ticks=0,fetcher=source))
        result=m.respond('яблоко');self.assertIn('fixture:article',result['sources'])
        self.assertEqual(result['relations'][0]['sentence'],'Яблоко является плодом яблони')
        path=Path(self.directory.name)/'test.cry';CrystalSnapshot.save(g,str(path));h=CrystalLattice();CrystalSnapshot.load(h,str(path))
        other=LanguageMembrane(h,data_dir=self.directory.name)
        self.assertEqual(other.respond('яблоко')['mode'],'Attributed')
    def test_readonly_answer_api(self):
        self.lab.session.membrane.inject_text('Яблоко это фрукт.')
        before=pickle.dumps(self.lab.session)
        data=self.client.get('/api/answer',query_string={'query':'Что такое яблоко?'}).json
        self.assertEqual(data['mode'],'Attributed');self.assertTrue(data['relations']);self.assertTrue(data['sources'])
        self.assertEqual(before,pickle.dumps(self.lab.session))
        self.assertEqual(self.client.get('/api/observations?instance=99').status_code,400)
        self.assertEqual(self.client.get('/api/answer',query_string={'query':'яблоко','mode':'verified'}).json['mode'],'Unknown')

    def test_verified_hypothesis_conflict_and_no_silent_upgrade(self):
        g=self.lab.population.active;m=self.lab.session.membrane
        g.connect('нагрев','расширение',70,EDGE_IS_A,CrystalReason('observation',source_label='verified experiment',metadata={'status':'confirmed'}))
        self.assertEqual(m.respond('нагрев','verified')['mode'],'Verified')
        a=g.get_or_create('a');b=g.get_or_create('b')
        g.propose_causal_link(a.id,b.id,CrystalReason('hypothesis',source_label='hypothesis fixture',confidence=.3))
        self.assertEqual(m.respond('a','hypothesis')['mode'],'Hypothesis')
        self.assertEqual(m.respond('a','verified')['mode'],'Unknown')
        reason=next(v for k,v in g.edge_meta.items() if len(k)==3)
        reason.metadata['status']='conflict'
        g.connect('нагрев','другое',60,EDGE_IS_A,CrystalReason('input',metadata={'status':'observed'}))
        self.assertEqual(m.respond('нагрев')['reason'],'conflict')

    def test_sleep_does_not_delete_incoming_class_target(self):
        g=self.lab.population.active
        g.connect('x','class-target',75,EDGE_IS_A,CrystalReason('input',metadata={'status':'observed'}))
        target=g.label_to_id['class-target'];g.resonators[target].energy=0
        g.defragment();self.assertIn(target,g.resonators)
        self.assertTrue(all(t in g.resonators for r in g.resonators.values() for t in r.connections))

    def test_conditional_strategy_transfer_to_heldout_subject(self):
        from imitation_learning import TrainingExample,answer,evaluate
        for question,reference in [('Что такое яблоко?','Яблоко — это плод яблони.'),('Что такое апельсин?','Апельсин — это цитрусовый плод.'),('Что такое груша?','Груша — это плод груши.')]:
            self.assertEqual(self.post('imitation/observation',{'question':question,'reference':reference}).status_code,200)
        held=TrainingExample('held','IS_A груша',('Груша — это плод груши.',),split='test')
        train,controls,analysis=prepare(self.lab.session,'Что такое яблоко?','Яблоко — это плод яблони.',[held])
        self.assertFalse(any('груша' in c.query for c in controls))
        test=TrainingExample('held','IS_A груша',('Груша — это плод груши.',),(('груша','IS_A','плод груши'),),split='test',evidence_mode='user_observed')
        learner=ImitationLearner(self.lab.population.active,controls)
        before=evaluate(answer(self.lab.population.active,test,learner.parameters),test)
        record=learner.train_on_example(train,Config(enabled=True,allow_user_observed=True))
        after=evaluate(answer(self.lab.population.active,test,learner.parameters),test)
        self.assertTrue(record['accepted'],record)
        self.assertEqual(before['semantic'],after['semantic']);self.assertEqual(before['form'],0);self.assertEqual(after['form'],1)
        self.assertEqual(self.lab.session.membrane.respond('груша','verified')['mode'],'Unknown')

    def test_parameter_positional_compatibility(self):
        from imitation_learning import Parameters
        p=Parameters(.1,1,.2,.3,.4)
        self.assertEqual((p.isa_priority,p.part_priority,p.cause_priority,p.is_a_construction),(.2,.3,.4,0))

    def test_sequential_source_reports_preserve_semantic_scope(self):
        from engine import ContextNode
        g=self.lab.population.active;m=self.lab.session.membrane
        def study(topic,text):
            list(research_steps(m,g,topic,ticks=0,fetcher=lambda q:dict(text=text,source='fixture:'+q,source_type='wiki')))
        study('яблоко','Яблоко — это фрукт.')
        study('апельсин','Апельсин — это цитрусовый плод.')
        reply=m.respond('яблоко')
        self.assertEqual(reply['mode'],'Attributed');self.assertIn('fixture:яблоко',reply['sources'])
        self.assertTrue(reply['relations'][0]['cross_source_context'])
        self.assertEqual(m.respond('яблоко','verified')['mode'],'Unknown')
        g.contexts['fiction']=ContextNode('fiction','global','dialog');g.active_context='fiction'
        study('дракон','Дракон — это персонаж.')
        study('яблоко','Яблоко — это артефакт.')
        g.active_context='global'
        self.assertEqual(m.respond('дракон')['reason'],'context_mismatch')
        self.assertIn('фрукт',m.respond('яблоко')['text'])
        self.assertEqual(g.contexts['wiki:яблоко'].parent,'global')
