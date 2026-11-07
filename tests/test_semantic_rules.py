import copy,json,tempfile,unittest
from pathlib import Path
from engine import CrystalLattice,CrystalReason,EDGE_PART_OF
from membrane import LanguageMembrane
from persistence import CrystalSnapshot
from research import ingest_observation
from semantic_rules import state,interpret,correct,questions,rollback,validate_rules

class RuleLearning(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.g=CrystalLattice();self.m=LanguageMembrane(self.g,data_dir=self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def test_active_transfer_rollback_and_unknown(self):
  for text in ['Двигатель состоит из деталей','Прибор состоит из модулей'] :self.m.inject_text(text)
  qs=questions(self.g);self.assertEqual(len(qs),2)
  for q in qs:
   a,b=q['sentence'].casefold().split(' состоит из ')
   result=correct(self.g,q['id'],'incorrect',b,'PART_OF',a)
  self.assertTrue(result['accepted'],result)
  f=interpret('Станция состоит из отсеков',state(self.g)['rules']);self.assertEqual(len(f),1);self.assertEqual(f[0]['subject'],'отсеков')
  self.assertFalse(questions(self.g));self.assertTrue(rollback(self.g));self.assertEqual(len(interpret('Станция состоит из отсеков',state(self.g)['rules'])),2)
 def test_unknown_is_not_negative_and_confirmed_survives(self):
  self.m.inject_text('Двигатель состоит из деталей');q=questions(self.g)[0];rules=copy.deepcopy(state(self.g)['rules'])
  correct(self.g,q['id'],'unknown');self.assertEqual(rules,state(self.g)['rules']);self.assertFalse(state(self.g)['controls'])
  self.g.connect('деталь','прибор',80,EDGE_PART_OF,CrystalReason('input',metadata={'status':'confirmed'}))
  ingest_observation(self.m,self.g,'деталь входит в состав прибор','source')
  self.assertTrue(any(r.metadata.get('status')=='confirmed' for r in self.g.edge_meta.values()))
 def test_persistence_and_duplicate_source(self):
  ingest_observation(self.m,self.g,'Прибор предназначен для измерений','source')
  before=len(self.g.relation_observations);ingest_observation(self.m,self.g,'Прибор предназначен для измерений','source');self.assertEqual(before,len(self.g.relation_observations))
  path=str(Path(self.temp.name)/'state.cry');CrystalSnapshot.save(self.g,path);other=CrystalLattice();CrystalSnapshot.load(other,path)
  self.assertEqual(state(self.g),state(other))
 def test_language_is_safe_and_multitype(self):
  rules=state(self.g)['rules'];bad=copy.deepcopy(rules);bad[0]['python']='import os'
  with self.assertRaises(ValueError):validate_rules(bad)
  for text,rel in [('Датчик предназначен для контроля','GOAL'),('Модуль аналогичен блоку','ANALOG'),('Сбой возникает вследствие перегрева','EFFECT'),('Запуск требует условия готовности','COND'),('Список за исключением заголовка','EXCEPT')]:
   frames=interpret(text,rules);self.assertEqual(frames[0]['relation'],rel)
  self.m.inject_text('Нагрев не вызывает плавление');self.assertTrue(state(self.g)['frames'][-1]['negated'])
 def test_api_readonly_and_correct(self):
  import pickle
  from crystal_lab import create_app
  app=create_app(self.temp.name);lab=app.extensions['lab'];client=app.test_client();headers={'X-Lab-Token':lab.token}
  before=pickle.dumps(lab.session);self.assertEqual(client.get('/api/semantic-learning').status_code,200);self.assertEqual(before,pickle.dumps(lab.session))
  lab.session.membrane.inject_text('Двигатель состоит из деталей');q=client.get('/api/semantic-learning').json['questions'][0]
  result=client.post('/api/semantic-learning/correct',json={'id':q['id'],'response':'unknown'},headers=headers)
  self.assertEqual(result.status_code,200);self.assertFalse(result.json['accepted'])

 def test_registered_lexeme_forms_are_exact(self):
  from semantic_space import SemanticSpace,Lexeme
  from structured_observations import candidate_labels
  self.g.semantic_space=SemanticSpace()
  for word,forms in [('яблоко',('яблока','яблоки')),('фрукт',('фрукта','фруктом')),('слово',('слова','словом'))]:
   self.g.get_or_create(word);self.g.semantic_space.lexemes[word]=Lexeme(word,word,forms)
   for form in forms:self.assertEqual(candidate_labels(self.m,form),[word])
  self.assertEqual(candidate_labels(self.m,'словарь'),[])
