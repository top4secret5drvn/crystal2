import tempfile, unittest
from pathlib import Path
from engine import CrystalLattice, CrystalReason
from persistence import CrystalSnapshot

class Snapshots(unittest.TestCase):
    def test_full_and_delta_keep_hypotheses(self):
        with tempfile.TemporaryDirectory() as d:
            g=CrystalLattice();g.connect('a','b',weight=90)
            a=g.label_to_id['a']; g.add_hypothesis(a,CrystalReason('hypothesis',metadata={'training_examples':['x']}))
            full=str(Path(d)/'base.cry');delta=str(Path(d)/'delta.cdt')
            CrystalSnapshot.save(g,full);cache=CrystalSnapshot.generate_state_cache(g)
            g.resonators[a].energy+=10; g.reject_hypothesis(a)
            CrystalSnapshot.save_delta(g,delta,cache)
            h=CrystalLattice(); self.assertTrue(CrystalSnapshot.load(h,full))
            self.assertIn(a,h.hypotheses)
            self.assertEqual(h.resonators[a].connections,g.resonators[a].connections)
            self.assertTrue(CrystalSnapshot.load(h,delta,cache))
            self.assertEqual(h.resonators[a].connections,g.resonators[a].connections)
            self.assertEqual(h.resonators[a].belief_status,'defeated')
            self.assertNotIn(a,h.hypotheses)

    def test_energy_only_delta_preserves_edges(self):
        with tempfile.TemporaryDirectory() as d:
            g=CrystalLattice();g.connect('a','b',weight=90)
            full=str(Path(d)/'base.cry');delta=str(Path(d)/'delta.cdt')
            CrystalSnapshot.save(g,full);cache=CrystalSnapshot.generate_state_cache(g)
            a=g.label_to_id['a'];g.resonators[a].energy+=10
            CrystalSnapshot.save_delta(g,delta,cache)
            h=CrystalLattice();CrystalSnapshot.load(h,full);CrystalSnapshot.load(h,delta,cache)
            self.assertEqual(g.resonators[a].connections,h.resonators[a].connections)

    def test_legacy_files(self):
        for name in ('bugfix2.cry','bugfix_test.cry'):
            g=CrystalLattice();self.assertTrue(CrystalSnapshot.load(g,str(Path(__file__).resolve().parents[1]/name)))
            self.assertTrue(g.resonators)
            self.assertTrue(all(t in g.resonators for r in g.resonators.values() for t in r.connections))
