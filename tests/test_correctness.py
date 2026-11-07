import unittest
from engine import CrystalLattice, CrystalReason, HDCEncoder, EDGE_SYNTAGM, unpack_edge
from cognitive_modules import BeliefRevisionEngine, Reason

class Correctness(unittest.TestCase):
    def test_correlation_is_not_causation(self):
        g = CrystalLattice()
        g.connect('rain', 'umbrella', weight=90, edge_type=EDGE_SYNTAGM)
        for r in g.resonators.values(): r.energy = 900
        g.infer_causal_links()
        a,b = g.label_to_id['rain'],g.label_to_id['umbrella']
        self.assertEqual(unpack_edge(g.resonators[a].connections[b])[1], EDGE_SYNTAGM)

    def test_revision_restores_defeated(self):
        e=BeliefRevisionEngine()
        e.add_belief('p(x)', 1, Reason('input', confidence=.6, metadata={'источник':'a'}))
        e.add_belief('p(x)', -1, Reason('input', confidence=.9, metadata={'источник':'b'}))
        e.set_source_reliability('b', .1)
        self.assertEqual(e.query('p(x)')[0], 1)

    def test_equal_conflict_is_not_unknown(self):
        e=BeliefRevisionEngine()
        self.assertIsNone(e.query('missing'))
        e.add_belief('p',1,Reason('input'))
        e.add_belief('p',-1,Reason('input'))
        self.assertEqual(e.query('p')[1].kind,'противоречие')

    def test_contexts_and_unknown(self):
        e=BeliefRevisionEngine()
        e.add_belief('p',1,Reason('input', confidence=.4,metadata={'контекст':'a'}))
        e.add_belief('p',-1,Reason('input',metadata={'контекст':'b'}))
        e.add_belief('p',0,Reason('input',metadata={'контекст':'a'}))
        self.assertEqual(e.query('p','a')[0],1)
        self.assertEqual(e.query('p','b')[0],-1)

    def test_hd_operations(self):
        e=HDCEncoder(); a,b=e.encode('a'),e.encode('b')
        self.assertEqual(a,HDCEncoder().encode(b'a'))
        self.assertEqual(e.bind(e.bind(a,b),b),a)
        self.assertEqual(e.invert(e.invert(a)),a)
        self.assertAlmostEqual(e.similarity(a,e.invert(a)),0)
        self.assertTrue(.45<e.similarity(a,b)<.55)
        self.assertTrue(.45<e.bundle([a,e.invert(a)]).bit_count()/e.DIM<.55)



class Hypotheses(unittest.TestCase):
    def test_structural_support_never_confirms(self):
        g=CrystalLattice()
        for name in ('a','b'):
            for i in range(4): g.connect(name, 'root:p'+str(i),weight=90)
        for r in g.resonators.values(): r.energy=200
        g.discover_abstractions()
        abstractions=[r for r in g.resonators.values() if r.label.startswith('mdl:')]
        self.assertTrue(abstractions)
        self.assertTrue(all(r.belief_status=='hypothesis' for r in abstractions))
        before=len(g.resonators); g.discover_abstractions()
        self.assertEqual(before,len(g.resonators))

    def test_heldout_lifecycle(self):
        g=CrystalLattice(); r=g.get_or_create('mdl:a+b')
        g.add_hypothesis(r.id,CrystalReason('hypothesis',metadata={'training_examples':['train']}))
        with self.assertRaises(ValueError): g.evaluate_abstraction(r.id,'train',True,True,independent=True)
        self.assertEqual(g.evaluate_abstraction(r.id,'x',True,False,independent=True),'defeated')
        self.assertEqual(g.evaluate_abstraction(r.id,'x',True,True,independent=True),'hypothesis')
        self.assertEqual(g.evaluate_abstraction(r.id,'y',True,True,independent=True),'confirmed')

    def test_causal_evidence(self):
        g=CrystalLattice(); a=g.get_or_create('a');b=g.get_or_create('b')
        key=g.propose_causal_link(a.id,b.id,CrystalReason('hypothesis'))
        with self.assertRaises(ValueError):
            g.evaluate_causal_link(key,'obs',intervention=False,temporal_order=True,replicated=True,outcome=True)
        self.assertEqual(g.evaluate_causal_link(key,'experiment',intervention=True,temporal_order=True,replicated=True,outcome=True),'confirmed')
        self.assertEqual(g.evaluate_causal_link(key,'counterexample',intervention=True,temporal_order=True,replicated=True,outcome=False),'defeated')

    def test_retraction_keeps_history(self):
        e=BeliefRevisionEngine()
        for value,conf,source in [(1,.5,'a'),(-1,.9,'b')]:
            e.add_belief('p',value,Reason('input',confidence=conf,metadata={'источник':source}))
        e.retract_source('b')
        self.assertEqual(e.query('p')[0],1)
        self.assertEqual(len(e.beliefs['p']),2)

class LatticeBeliefs(unittest.TestCase):
    def test_weaker_objection_preserves_winner(self):
        g=CrystalLattice();r=g.get_or_create('belief')
        g.set_belief_status(r.id,'observed',.9,CrystalReason('input',source_type='user',confidence=.9))
        g.resolve_belief_conflict(r.id,'defeated',.1,CrystalReason('input',source_type='wiki',confidence=.1))
        self.assertEqual(r.belief_status,'observed')
        self.assertEqual(len(g.belief_history[r.id]),2)

    def test_lattice_retraction_restores_previous(self):
        g=CrystalLattice();r=g.get_or_create('belief')
        g.set_belief_status(r.id,'observed',.5,CrystalReason('input',source_label='a',source_type='user',confidence=.5))
        g.resolve_belief_conflict(r.id,'defeated',.9,CrystalReason('input',source_label='b',source_type='user',confidence=.9))
        g.retract_belief_source(r.id,'b')
        self.assertEqual(r.belief_status,'observed')
        self.assertEqual(r.belief_reason.source_label,'a')

if __name__=='__main__': unittest.main()
