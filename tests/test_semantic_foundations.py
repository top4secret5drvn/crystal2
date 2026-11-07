import os,tempfile,unittest
from engine import *
from membrane import LanguageMembrane
from persistence import CrystalSnapshot

class Foundations(unittest.TestCase):
    def test_multitype(self):
        g=CrystalLattice();g.connect('a','b',80,EDGE_IS_A);g.connect('a','b',90,EDGE_SYNTAGM)
        a=g.get_or_create('a');b=g.get_or_create('b')
        self.assertIn((b.id,80),a.get_connections_by_type(EDGE_IS_A))
        self.assertIn((b.id,90),a.get_connections_by_type(EDGE_SYNTAGM))
        self.assertFalse(b.get_connections_by_type(EDGE_IS_A))

    def test_epoch_identity(self):
        a=CrystalLattice();a.set_epoch('one');x=a.get_or_create('apple')
        b=CrystalLattice();b.set_epoch('two');y=b.get_or_create('apple')
        self.assertEqual(x.hdc_vector,y.hdc_vector)

    def test_sentence_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                for label in ('камень','гранит','река','вода'):g.get_or_create(m._resolve_label(label))
                m.inject_text('камень гранит. река вода.')
                a=g.get_or_create(m._resolve_label('гранит'));b=g.get_or_create(m._resolve_label('река'))
                self.assertNotIn(b.id,a.connections)
            finally:os.chdir(old)

    def test_syn_not_factual(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                g.connect('яблоко','камень',90)
                for r in g.resonators.values():r.energy=1000
                self.assertIn('Недостаточно',m.generate_text(['яблоко']))
            finally:os.chdir(old)

    def test_generation_needs_typed_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                reason=CrystalReason('observation',source_label='verified-example',metadata={'status':'confirmed'})
                g.connect('яблоко','фрукт',50,EDGE_IS_A,reason)
                g.connect('яблоко','камень',200,EDGE_SYNTAGM)
                self.assertIn('фрукт',m.generate_text(['яблоко']))
                self.assertNotIn('камень',m.generate_text(['яблоко']))
                self.assertIn('Ассоциативный',m.generate_text(['яблоко'],mode='associative'))
            finally:os.chdir(old)

    def test_negation_scoped_to_sentence(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                m.inject_text('кит не является рыбой. яблоко это фрукт.')
                self.assertTrue(g.relation_observations)
                self.assertTrue(any(r.get_connections_by_type(EDGE_IS_A) for r in g.resonators.values()))
                self.assertIn('Недостаточно',m.generate_text(['яблоко']))
            finally:os.chdir(old)

    def test_copula_is_observed_typed_relation(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                for word in ('яблоко','фрукт'):g.get_or_create(m._resolve_label(word))
                m.inject_text('яблоко является фрукт')
                self.assertTrue(any(r.get_connections_by_type(EDGE_IS_A) for r in g.resonators.values()))
                self.assertIn('Недостаточно',m.generate_text(['яблоко']))
            finally:os.chdir(old)

    def test_all_relation_types_coexist(self):
        g=CrystalLattice()
        kinds=(EDGE_SYNTAGM,EDGE_IS_A,EDGE_PART_OF,EDGE_CAUSE,EDGE_ANALOG,EDGE_EXCEPT)
        for kind in kinds:g.connect('a','b',60,kind)
        a,b=g.get_or_create('a'),g.get_or_create('b')
        for kind in kinds:self.assertIn((b.id,60),a.get_connections_by_type(kind))
        self.assertEqual(g.find_analogies('a'),[('b',60)])
        a.connections.remove_type(b.id,EDGE_CAUSE)
        self.assertEqual(len(list(a.connections.typed_items())),5)

    def test_factual_generation_respects_context(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                g.set_epoch('first')
                g.connect('яблоко','фрукт',60,EDGE_IS_A,CrystalReason('observation',context='first',metadata={'status':'confirmed'}))
                self.assertIn('фрукт',m.generate_text(['яблоко']))
                g.set_epoch('second')
                self.assertIn('Недостаточно',m.generate_text(['яблоко']))
            finally:os.chdir(old)
