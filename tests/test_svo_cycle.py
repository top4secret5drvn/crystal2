import tempfile,unittest
from pathlib import Path
from engine import CrystalLattice
from membrane import LanguageMembrane

class SVOCycle(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.g=CrystalLattice();self.m=LanguageMembrane(self.g,data_dir=self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def test_requested_cases(self):
  for sentence,question,answer in [('кот ест рыбу','что ест кот?','рыбу'),('кот ест рыбу','кто ест рыбу?','кот'),('солнце это звезда','что такое солнце?','звезда'),('солнце нагревает землю','что нагревает землю?','солнце')]:
   self.m.inject_text(sentence);r=self.m.respond(question)
   self.assertEqual(r['mode'],'Attributed',(sentence,r));self.assertIn(answer,r['text']);self.assertTrue(r['sources']);self.assertTrue(r['relations'])
 def test_negative_and_conflict(self):
  self.m.inject_text('кот не ест рыбу');self.assertEqual(self.m.respond('что ест кот?')['mode'],'Unknown')
  self.m.inject_text('кот ест рыбу');self.assertEqual(self.m.respond('что ест кот?')['reason'],'conflict')
 def test_new_words_and_missing(self):
  self.m.inject_text('мастер читает книгу');r=self.m.respond('что читает мастер?');self.assertEqual(r['mode'],'Attributed');self.assertIn('книгу',r['text'])
  self.assertEqual(self.m.respond('что строит мастер?')['mode'],'Unknown')
 def test_snapshot(self):
  from persistence import CrystalSnapshot
  from research import ingest_observation
  ingest_observation(self.m,self.g,'робот видит башню','local:test')
  p=str(Path(self.temp.name)/'s.cry');CrystalSnapshot.save(self.g,p);g=CrystalLattice();CrystalSnapshot.load(g,p)
  r=LanguageMembrane(g,data_dir=self.temp.name).respond('кто видит башню?');self.assertEqual(r['mode'],'Attributed');self.assertIn('local:test',r['sources'])
 def test_direct_check_modes_and_trace(self):
  self.m.inject_text('мастер читает книгу')
  r=self.m.respond('мастер читает книгу?');self.assertEqual(r['mode'],'Attributed');self.assertIn('Да',r['answer']);self.assertTrue(r['relations'][0]['provenance_id'])
  self.assertEqual(self.m.respond('что читает мастер?','verified')['mode'],'Unknown')
  self.assertEqual(self.m.respond('что читает мастер?','unknown')['mode'],'Unknown')
 def test_context_and_sleep(self):
  from engine import ContextNode
  self.g.contexts['fiction']=ContextNode('fiction','global','dialog');self.g.active_context='fiction'
  self.m.inject_text('робот открывает дверь');self.g.defragment()
  self.assertEqual(self.m.respond('что открывает робот?')['mode'],'Attributed')
  f=self.g.semantic_rules_state['frames'][0];self.assertIn(f['subject_id'],self.g.resonators);self.assertIn(f['object_id'],self.g.resonators)
  self.g.active_context='global';self.assertEqual(self.m.respond('что открывает робот?')['mode'],'Unknown')
 def test_normalization_and_unknown_grammar(self):
  self.m.inject_text('солнце нагревает землю')
  self.assertEqual(self.m.respond('кто нагревать земля?')['mode'],'Attributed')
  from svo_observations import lemma
  self.assertEqual(lemma('звёзды'),'звезда');self.assertEqual(lemma('звезды'),'звезда')
  stats=self.m.inject_text('сложный неизвестный оборот');self.assertEqual(stats.parse_errors,1)
  self.assertEqual(self.m.respond('кто изучает звезду?')['mode'],'Unknown')
 def test_cli_and_diagnostics(self):
  from command_session import CommandSession
  from engine import CrystalPopulation
  p=CrystalPopulation();p.crystals=[self.g];s=CommandSession(p,self.m)
  for c in ('ввод солнце нагревает землю','спроси что нагревает землю?','ответь'):
   result=s.execute(c);self.assertFalse(result.error,result.output)
  self.assertEqual(s.last_response['mode'],'Attributed');self.assertIn('солнце',s.last_response['text'])
  stats=self.m.inject_text('работник проверяет прибор');self.assertEqual(stats.semantic_frames,1);self.assertEqual(stats.semantic_relations,1)

 def test_distinct_sources_and_repetition_do_not_inflate_confidence(self):
  from research import ingest_observation
  for source in ('local:a','local:b','local:b'):ingest_observation(self.m,self.g,'мастер читает книгу',source)
  r=self.m.respond('что читает мастер?');self.assertEqual(set(r['sources']),{'local:a','local:b'});self.assertEqual(r['confidence'],.6)
  self.assertEqual(len([f for f in self.g.semantic_rules_state['frames'] if f['relation']=='SVO']),2)
