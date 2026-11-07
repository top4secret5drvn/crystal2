import tempfile,unittest
from engine import CrystalLattice
from membrane import LanguageMembrane

class TrialLearning(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.g=CrystalLattice();self.m=LanguageMembrane(self.g,data_dir=self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def test_trial_feedback_without_controls(self):
  from imitation_learning import ExperimentalLearningLoop
  self.g.connect('мирак','осколки',100);self.g.get_or_create('луны')
  loop=ExperimentalLearningLoop(self.m);attempt=loop.attempt('что поглощает мирак?')
  self.assertEqual(attempt['mode'],'Hypothesis');self.assertTrue(attempt['candidates'])
  result=loop.feedback('reference_answer','мирак поглощает луны')
  self.assertEqual(result['phase'],'provisional');self.assertTrue(result['mutations'])
  self.assertNotEqual(self.m.respond('что поглощает мирак?','verified')['mode'],'Verified')
  self.assertEqual(loop.attempt('что поглощает мирак?')['selected']['object'],'луны')
 def fixture(self):
  from engine import EDGE_IS_A
  for name in ('мирак','новал','третак'):self.g.connect(name,'классА',70,EDGE_IS_A)
  for name in ('луны','споры','камни'):self.g.connect(name,'классБ',70,EDGE_IS_A)
  for name in ('мирак','новал','третак'):self.g.connect(name,'ошибка',100)
 def test_full_cycle_and_transfer(self):
  from imitation_learning import ExperimentalLearningLoop
  self.fixture();loop=ExperimentalLearningLoop(self.m)
  before=loop.attempt('что поглощает мирак?');self.assertEqual(before['selected']['object'],'ошибка')
  rejected=loop.feedback('incorrect');self.assertTrue(rejected['accepted'])
  corrected=loop.feedback('reference_answer','мирак поглощает луны');self.assertTrue(corrected['accepted'])
  self.assertEqual(loop.attempt('что поглощает мирак?')['selected']['object'],'луны')
  # New targets: each validation fixture offers just one class-compatible choice.
  self.g.connect('новал','споры',10)
  # Existing class-compatible objects are legitimate competing hypotheses.
  held=loop.attempt('что поглощает новал?');self.assertEqual(held['selected']['object'],'споры')
  self.assertTrue(all(h['status']=='hypothesis' for h in held['candidates']))
  self.assertNotEqual(self.m.respond('что поглощает новал?','verified')['mode'],'Verified')
 def test_rollback_and_evidence_unchanged(self):
  import pickle
  from imitation_learning import ExperimentalLearningLoop
  self.g.connect('a','wrong',100);self.g.get_or_create('right');loop=ExperimentalLearningLoop(self.m)
  evidence=pickle.dumps((self.g.resonators,self.g.edge_meta,self.g.relation_observations))
  loop.attempt('что делает a?');loop.feedback('reference_answer','a делает right')
  self.assertEqual(evidence,pickle.dumps((self.g.resonators,self.g.edge_meta,self.g.relation_observations)))
  self.assertTrue(loop.rollback());self.assertFalse(loop.status()['model']['edges'])
 def test_unknown_budget_and_snapshot(self):
  import pickle
  from imitation_learning import ExperimentalLearningLoop
  from persistence import CrystalSnapshot
  from pathlib import Path
  self.g.connect('a','b');loop=ExperimentalLearningLoop(self.m);loop.attempt('что делает a?')
  model=pickle.dumps(loop.status()['model']);loop.feedback('unknown');self.assertEqual(model,pickle.dumps(loop.status()['model']))
  path=str(Path(self.temp.name)/'trial.cry');CrystalSnapshot.save(self.g,path);other=CrystalLattice();CrystalSnapshot.load(other,path)
  restored=ExperimentalLearningLoop(LanguageMembrane(other,data_dir=self.temp.name));self.assertEqual(loop.status(),restored.status())
  for _ in range(7):loop.attempt('что делает a?')
  with self.assertRaises(ValueError):loop.attempt('что делает a?')
 def test_cli(self):
  from command_session import CommandSession
  from engine import CrystalPopulation
  self.g.connect('мирак','ошибка',100);self.g.get_or_create('луны')
  pop=CrystalPopulation();pop.crystals=[self.g];session=CommandSession(pop,self.m)
  for c in ('спроси что поглощает мирак?','попробуй','неверно','эталон мирак поглощает луны','повтори','обучение статус','обучение откат'):
   result=session.execute(c);self.assertFalse(result.error,(c,result.output))
 def test_contradictory_feedback_never_confirms_facts(self):
  from imitation_learning import ExperimentalLearningLoop
  self.fixture();loop=ExperimentalLearningLoop(self.m);loop.attempt('что поглощает мирак?')
  loop.feedback('reference_answer','мирак поглощает луны');loop.attempt('что поглощает мирак?');loop.feedback('correct')
  with self.assertRaises(ValueError):loop.feedback('correct')
  loop.feedback('incorrect');self.assertEqual(loop.status()['phase'],'provisional')
  self.assertTrue(all(e['status']=='hypothesis' for e in loop.status()['model']['edges']))
  self.assertNotEqual(self.m.respond('что поглощает мирак?','verified')['mode'],'Verified')
 def test_context_and_wrong_reference_are_safe(self):
  import copy
  from imitation_learning import ExperimentalLearningLoop
  self.g.connect('a','b');loop=ExperimentalLearningLoop(self.m);loop.attempt('что делает a?');before=copy.deepcopy(loop.status())
  with self.assertRaises(ValueError):loop.feedback('reference_answer','other делает b')
  self.assertEqual(before,loop.status())
  self.g.set_epoch('other')
  with self.assertRaises(ValueError):loop.feedback('incorrect')
 def test_sequential_tasks_keep_prior_corrected_task(self):
  from imitation_learning import ExperimentalLearningLoop
  self.fixture();loop=ExperimentalLearningLoop(self.m);loop.attempt('что поглощает мирак?');loop.feedback('reference_answer','мирак поглощает луны')
  loop.attempt('что отражает новал?');result=loop.feedback('reference_answer','новал отражает споры')
  self.assertTrue(result['accepted']);self.assertEqual(result['independent_controls'],1)
  self.assertEqual(result['before'][1:],result['after'][1:])
  self.assertEqual(loop.attempt('что поглощает мирак?')['selected']['object'],'луны')

 def test_empty_memory_can_adapt_after_reference(self):
  from imitation_learning import ExperimentalLearningLoop
  loop=ExperimentalLearningLoop(self.m);self.assertEqual(loop.attempt('что делает аран?')['mode'],'Unknown')
  record=loop.feedback('reference_answer','аран делает бран');self.assertTrue(record['accepted'])
  self.assertEqual(record['reference_text'],'аран делает бран');self.assertEqual(record['phase'],'provisional')
  retry=loop.attempt('что делает аран?');self.assertEqual(retry['mode'],'Hypothesis');self.assertEqual(retry['selected']['object'],'бран')
  self.assertEqual(retry['selected']['nodes'],[])
 def test_existing_missing_validation_gate_is_separate_from_experiment(self):
  from imitation_learning import ImitationLearner,TrainingExample,Config,ExperimentalLearningLoop
  self.g.connect('a','wrong',100);self.g.get_or_create('b')
  example=TrainingExample('baseline','IS_A a',('A — это b.',),(('a','IS_A','b'),),correction_confidence=1)
  baseline=ImitationLearner(self.g).train_on_example(example,Config(enabled=True))
  self.assertFalse(baseline['accepted']);self.assertEqual(baseline['reason'],'missing_validation_or_evidence')
  loop=ExperimentalLearningLoop(self.m);loop.attempt('что такое a?')
  adapted=loop.feedback('reference_answer','a — это b')
  self.assertTrue(adapted['accepted']);self.assertEqual(adapted['phase'],'provisional')
