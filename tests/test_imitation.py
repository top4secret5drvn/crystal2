import pickle,random,unittest
from engine import CrystalLattice,CrystalReason,EDGE_IS_A,EDGE_PART_OF
from imitation_learning import *


def fixture():
    g=CrystalLattice()
    for name in ('яблоко','апельсин','молоток','модуль'):
        g.connect(name,'предмет',70,EDGE_IS_A,CrystalReason('observation',confidence=.9,metadata={'status':'confirmed'}))
        g.connect(name,'система',90,EDGE_PART_OF,CrystalReason('observation',confidence=.9,metadata={'status':'confirmed'}))
    def example(name,split):
        return TrainingExample(name,'IS_A '+name,(name.capitalize()+' является предметом.',),
                    ((name,'IS_A','предмет'),),correction_source='fixture',correction_confidence=1,split=split)
    return g,example('яблоко','train'),example('апельсин','validation'),example('молоток','test')

class Imitation(unittest.TestCase):
    def test_baseline_and_transfer(self):
        g,train,val,test=fixture();learner=ImitationLearner(g,[val])
        self.assertEqual(answer(g,train).relation[1],'PART_OF')
        before=pickle.dumps(g);rng=random.getstate()
        first=learner.train_on_example(train,Config(enabled=True))
        self.assertTrue(first['accepted'])
        self.assertEqual(first['diagnosis']['kind'],'C')
        second=learner.train_on_example(train,Config(enabled=True))
        self.assertTrue(second['accepted'])
        result=evaluate(answer(g,test,learner.parameters),test)
        self.assertEqual(result['semantic'],1);self.assertEqual(result['form'],1)
        self.assertEqual(before,pickle.dumps(g));self.assertEqual(rng,random.getstate())
        self.assertTrue(learner.rollback());self.assertTrue(learner.rollback())
        self.assertEqual(learner.parameters,Parameters())

    def test_missing_unannotated_disabled_and_budget(self):
        g,t,v,_=fixture();learner=ImitationLearner(g,[v])
        self.assertFalse(learner.train_on_example(t)['accepted'])
        missing=replace(t,id='missing',relations=(('яблоко','CAUSE','тепло'),))
        self.assertEqual(diagnose(answer(g,missing),missing)['kind'],'D')
        self.assertFalse(learner.train_on_example(missing,Config(enabled=True))['accepted'])
        self.assertFalse(learner.train_on_example(replace(t,relations=()),Config(enabled=True))['accepted'])
        self.assertFalse(learner.train_on_example(t,Config(enabled=True,max_evaluations=1))['accepted'])
        with self.assertRaises(ValueError):learner.train_on_example(replace(t,split='test'))

    def test_rejected_candidates_and_state(self):
        g,t,v,_=fixture();l=ImitationLearner(g,[v])
        # Same ambiguous query requires conflicting choices across train/control.
        t=replace(t,query='яблоко')
        v=replace(v,query='апельсин',relations=(('апельсин','PART_OF','система'),),references=('Апельсин входит в состав система.',))
        l=ImitationLearner(g,[v]);before=l.parameters
        r=l.train_on_example(t,Config(enabled=True))
        self.assertFalse(r['accepted']);self.assertEqual(before,l.parameters)
        self.assertTrue(r['attempts']);self.assertLessEqual(r['evaluations'],300)
        restored=ImitationLearner(g,[v],json.loads(json.dumps(l.state())))
        self.assertEqual(restored.parameters,l.parameters)

    def test_equivalence_is_not_text_similarity(self):
        g,t,v,_=fixture()
        t=replace(t,references=('Яблоко — предмет.', 'Яблоко является предметом.'))
        a=answer(g,t,Parameters(request_weight=.25,is_a_form=1))
        self.assertEqual(evaluate(a,t)['semantic'],1)
        fake=replace(a,relation=('яблоко','PART_OF','система'),text=t.references[1])
        self.assertEqual(evaluate(fake,t)['semantic'],0)

    def test_opt_in_snapshot_and_history_isolation(self):
        import tempfile
        from pathlib import Path
        from membrane import LanguageMembrane
        from persistence import CrystalSnapshot
        g,t,v,test=fixture();l=ImitationLearner(g,[v])
        for _ in range(2):l.train_on_example(t,Config(enabled=True))
        exported=l.state();exported['history'].clear();self.assertTrue(l.history)
        g.imitation_state=l.state()
        self.assertIn('предметом',LanguageMembrane(g).generate_text(['IS_A молоток'],mode='imitation'))
        self.assertIn('система',LanguageMembrane(g).generate_text(['молоток']))
        with tempfile.TemporaryDirectory() as d:
            filename=str(Path(d)/'imitation.cry');CrystalSnapshot.save(g,filename)
            h=CrystalLattice();self.assertTrue(CrystalSnapshot.load(h,filename))
            restored=ImitationLearner(h,[v],h.imitation_state)
            self.assertEqual(restored.parameters,l.parameters)
            self.assertEqual(answer(h,test,restored.parameters).relation,test.relations[0])
            self.assertTrue(restored.rollback())
        self.assertIsNone(CrystalLattice().imitation_state)

    def test_actual_baseline_and_error_categories(self):
        from membrane import LanguageMembrane
        g,t,v,_=fixture()
        self.assertIn('система',LanguageMembrane(g).generate_text(['яблоко']))
        self.assertEqual(diagnose(answer(g,t,Parameters(request_weight=.25)),t)['kind'],'A')
        g.connect('яблоко','предмет',99,EDGE_PART_OF,CrystalReason('observation',metadata={'status':'confirmed'}))
        self.assertEqual(diagnose(answer(g,t),t)['kind'],'B')
        g2=CrystalLattice()
        for kind in (EDGE_IS_A,EDGE_PART_OF):g2.connect('x','y',70,kind,CrystalReason('observation',metadata={'status':'confirmed'}))
        e=TrainingExample('ambiguous','x',(),(('x','IS_A','y'),),correction_source='fixture',correction_confidence=1)
        self.assertEqual(diagnose(answer(g2,e),e)['kind'],'E')

    def test_deterministic_seed_and_no_control_leak(self):
        g,t,v,_=fixture();a=ImitationLearner(g,[v]);b=ImitationLearner(g,[v])
        self.assertEqual(a.train_on_example(t,Config(enabled=True,seed=29)),b.train_on_example(t,Config(enabled=True,seed=29)))
        with self.assertRaises(ValueError):ImitationLearner(g,[replace(v,split='test')])
        with self.assertRaises(ValueError):ImitationLearner(g,[replace(v,id=t.id)]).train_on_example(t,Config(enabled=True))

    def test_new_target_noun_transfer(self):
        g,t,v,test=fixture()
        # The noun undergoing inflection is unseen in training, not just the subject.
        g.connect('молоток','ресурс',95,EDGE_IS_A,CrystalReason('observation',confidence=.9,metadata={'status':'confirmed'}))
        test=replace(test,relations=(('молоток','IS_A','ресурс'),),references=('Молоток является ресурсом.',))
        l=ImitationLearner(g,[v])
        for _ in range(2):l.train_on_example(t,Config(enabled=True))
        result=evaluate(answer(g,test,l.parameters),test)
        self.assertEqual(result['semantic'],1);self.assertEqual(result['form'],1)
