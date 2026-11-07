import copy,json,pickle,tempfile,threading,unittest
from pathlib import Path
from crystal_lab import create_app
from engine import CrystalLattice,CrystalReason,EDGE_IS_A,EDGE_SYNTAGM
from persistence import CrystalSnapshot
from command_session import CommandSession
from semantic_space import SemanticSpace,Evidence
from imitation_learning import TrainingExample
from dataclasses import asdict

class LabAPI(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.app=create_app(self.root);self.client=self.app.test_client();self.lab=self.app.extensions['lab']
        self.headers={'X-Lab-Token':self.lab.token}
    def tearDown(self):self.directory.cleanup()
    def post(self,path,data):return self.client.post('/api/'+path,json=data,headers=self.headers)

    def test_read_is_pure_and_command_shared(self):
        before=pickle.dumps(self.lab.population)
        for route in ('status','graph','instances','history','geometry','training','snapshots'):
            self.assertEqual(self.client.get('/api/'+route).status_code,200)
        self.assertEqual(before,pickle.dumps(self.lab.population))
        result=self.post('commands',{'command':'думай 2'})
        self.assertEqual(result.status_code,200,result.json);self.assertEqual(self.lab.population.active.tick_count,2)
        session=CommandSession();session.execute('думай 2')
        self.assertEqual(session.population.active.tick_count,self.lab.population.active.tick_count)
        self.assertTrue(list((self.root/'backups').glob('*.cry')))

    def test_snapshot_copy_overwrite_and_paths(self):
        source=self.root/'original.cry';g=CrystalLattice();g.get_or_create('source');CrystalSnapshot.save(g,str(source));original=source.read_bytes()
        self.assertEqual(self.post('snapshots/load',{'name':'original.cry'}).status_code,200)
        self.post('commands',{'command':'ввод Новый текст'})
        self.assertEqual(original,source.read_bytes())
        self.assertEqual(self.post('snapshots/save',{'name':'original.cry'}).status_code,400)
        self.assertEqual(original,source.read_bytes())
        self.assertEqual(self.post('snapshots/save',{'name':'original.cry','confirm_overwrite':True}).status_code,200)
        for name in ('../x.cry','C:\\x.cry','/tmp/x.cry','backups/x.cry'):
            self.assertEqual(self.post('snapshots/save',{'name':name}).status_code,400)
        bad=self.root/'bad.cry';bad.write_bytes(b'garbage')
        before=pickle.dumps(self.lab.population)
        self.assertEqual(self.post('snapshots/load',{'name':'bad.cry'}).status_code,400)
        self.assertEqual(before,pickle.dumps(self.lab.population))

    def test_security_and_limits(self):
        self.assertEqual(self.client.post('/api/commands',json={'command':'думай 1'}).status_code,403)
        self.assertEqual(self.client.get('/api/status',headers={'Host':'evil.test'}).status_code,403)
        self.assertEqual(self.client.post('/api/commands',json={'command':'думай 1'},headers={**self.headers,'Origin':'https://evil.test'}).status_code,403)
        for text in ('__import__("os")','читай ../secret','изучи секрет','сохрани ../x','думай 99999','ввод '+'x'*2001):
            self.assertEqual(self.post('commands',{'command':text}).status_code,400)
        before=self.lab.population.active.tick_count
        self.assertEqual(self.post('commands',{'command':'думай 0'}).status_code,400)
        self.assertEqual(before,self.lab.population.active.tick_count)

    def test_graph_boundaries_and_provenance(self):
        g=self.lab.population.active
        r=CrystalReason('observation',source_label='test',metadata={'status':'confirmed'})
        g.connect('apple','fruit',70,EDGE_IS_A,r);g.connect('apple','fruit',30,EDGE_SYNTAGM)
        for i in range(400):g.get_or_create('item'+str(i))
        data=self.client.get('/api/graph?limit=20&services=true').json
        self.assertLessEqual(len(data['nodes']),20);self.assertTrue(data['partial'])
        a=g.label_to_id['apple'];b=g.label_to_id['fruit']
        data=self.client.get('/api/relations/'+f'{a}/{b}/IS_A').json
        self.assertEqual(data['reason']['source'],'test')
        self.assertEqual(self.client.get(f'/api/nodes/{a}').status_code,200)

    def test_learning_import_run_and_rollback(self):
        g=self.lab.population.active;rows=[]
        for name,split in [('apple','train'),('orange','validation')]:
            g.connect(name,'предмет',70,EDGE_IS_A,CrystalReason('observation',confidence=.9,metadata={'status':'confirmed'}))
            rows.append(asdict(TrainingExample(name,'IS_A '+name,(name.capitalize()+' является предметом.',),((name,'IS_A','предмет'),),correction_source='test',correction_confidence=1,split=split)))
        self.assertEqual(self.post('training/examples',{'examples':rows}).status_code,200)
        before=[(r.id,r.hdc_vector,list(r.connections.typed_items())) for r in g.resonators.values()]
        result=self.post('training/run',{'id':'apple'})
        self.assertEqual(result.status_code,200,result.json);self.assertTrue(result.json['accepted'])
        self.assertEqual(before,[(r.id,r.hdc_vector,list(r.connections.typed_items())) for r in g.resonators.values()])
        self.assertEqual(self.post('training/run',{'id':'orange'}).status_code,400)
        self.assertEqual(self.post('training/rollback',{}).status_code,200)

    def test_actual_diff_and_population_selection(self):
        g=self.lab.population.active;g.get_or_create('existing').energy=200
        result=self.post('commands',{'command':'думай 1'})
        self.assertEqual(result.status_code,200,result.json)
        event=self.client.get('/api/history').json[-1]
        self.assertGreater(event['diff']['totals']['energy_changes'],0)
        self.lab.population.crystals.append(CrystalLattice())
        self.assertEqual(self.client.get('/api/status?instance=1').json['nodes'],1)
        self.assertEqual(self.lab.population.active_index,0)
        self.assertEqual(self.post('instances/select',{'id':1}).status_code,200)
        self.assertIsNone(self.client.get('/api/history').json[-1]['diff'])

    def test_geometry_is_real_and_untrained_remains_unknown(self):
        s=SemanticSpace(8,17);s.add_world('world',('entity','class'),'test')
        ids={word:s.add_sense(word,word,role,['world']) for word,role in [('a','entity'),('b','entity'),('u','entity'),('c','class')]}
        for word in ('a','b'):s.observe(Evidence(word,ids[word],ids['c'],'IS_A','world','test',confirmed=True))
        s.train('world');self.lab.population.active.semantic_space=s
        response=self.client.get('/api/geometry?world=world').json
        self.assertEqual(response['dimensions'],8)
        self.assertEqual(next(r for r in response['senses'] if r['label']=='u')['coordinates'],None)
        self.assertEqual(next(r for r in response['senses'] if r['label']=='a')['coordinates'],list(s.position(ids['a'],'world')))

    def test_conflicting_operation_refused(self):
        ready=threading.Event();release=threading.Event()
        def hold():
            with self.lab.lock:ready.set();release.wait(5)
        worker=threading.Thread(target=hold);worker.start();ready.wait(5)
        try:self.assertEqual(self.post('commands',{'command':'думай 1'}).status_code,409)
        finally:release.set();worker.join()

    def test_workspace_lease_and_reserved_names(self):
        from crystal_lab.lease import WorkspaceLease
        with WorkspaceLease(self.root):
            with self.assertRaises(RuntimeError):WorkspaceLease(self.root)
        with WorkspaceLease(self.root):pass
        for name in ('CON.cry','NUL.cry','LPT1.cry'):
            self.assertEqual(self.post('snapshots/save',{'name':name}).status_code,400)

    def test_command_error_restores_actual_memory(self):
        from unittest.mock import patch
        before=pickle.dumps(self.lab.population)
        original=CommandSession.execute
        def broken(session,text):
            session.population.active.get_or_create('uncommitted')
            raise ValueError('synthetic failure')
        with patch.object(CommandSession,'execute',broken):
            self.assertEqual(self.post('commands',{'command':'ввод x'}).status_code,400)
        self.assertEqual(before,pickle.dumps(self.lab.population))
        self.assertTrue(self.client.get('/api/history').json[-1]['error'])

    def test_symlink_refused(self):
        target=self.root.parent/'outside-crystal-test.cry';target.write_bytes(b'outside')
        try:
            try:(self.root/'link.cry').symlink_to(target)
            except (OSError,NotImplementedError):self.skipTest('Symlinks unavailable for current Windows user')
            self.assertEqual(self.post('snapshots/load',{'name':'link.cry'}).status_code,400)
            self.assertEqual(self.post('snapshots/save',{'name':'link.cry','confirm_overwrite':True}).status_code,400)
            self.assertEqual(target.read_bytes(),b'outside')
        finally:target.unlink(missing_ok=True)

    def test_confirmed_filter_and_stale_browser(self):
        g=self.lab.population.active
        g.connect('known','class',70,EDGE_IS_A,CrystalReason('observation',source_label='fixture',metadata={'status':'confirmed'}))
        result=self.client.get('/api/graph?scope=confirmed').json
        self.assertEqual(len(result['nodes']),2)
        self.assertEqual(len(result['edges']),1)
        self.lab.population.crystals.append(CrystalLattice());self.lab.population.active_index=1
        result=self.post('commands',{'command':'думай 1','_instance':0})
        self.assertEqual(result.status_code,400)
        self.assertEqual(self.lab.population.active.tick_count,0)

    def test_real_transfer_event(self):
        population=self.lab.population;source=CrystalLattice()
        source.connect('anchor','neighbor',80);source.get_or_create('anchor').energy=400
        source.is_dormant=True;population.crystals.append(source)
        self.assertTrue(population.recall_from_hive_mind('anchor'))
        event=population.transfer_events[-1]
        self.assertEqual(event['source'],1);self.assertEqual(event['target'],0)
        self.assertGreater(event['nodes'],0);self.assertEqual(event['anchor'],'anchor')
