import os,tempfile,unittest
from engine import CrystalLattice, EDGE_CAUSE, unpack_edge
from membrane import LanguageMembrane

class Membrane(unittest.TestCase):
    def test_causal_text_is_observation(self):
        with tempfile.TemporaryDirectory() as d:
            previous=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                m.inject_text('дождь вызывает наводнение')
                self.assertFalse(any(unpack_edge(e)[1]==EDGE_CAUSE for r in g.resonators.values() for e in r.connections.values()))
                self.assertTrue(g.causal_hypotheses)
            finally: os.chdir(previous)

    def test_negation_does_not_defeat_class(self):
        with tempfile.TemporaryDirectory() as d:
            previous=os.getcwd()
            try:
                os.chdir(d);g=CrystalLattice();m=LanguageMembrane(g)
                m.inject_text('кит не является рыбой')
                self.assertFalse(any(r.belief_status=='defeated' for r in g.resonators.values()))
            finally: os.chdir(previous)
