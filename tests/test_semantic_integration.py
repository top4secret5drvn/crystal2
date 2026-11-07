import unittest,tempfile
from pathlib import Path
from engine import *
from persistence import CrystalSnapshot
from test_semantic_space import fixture

class Integration(unittest.TestCase):
    def test_modes_do_not_assert_facts(self):
        space,ids=fixture();space.train('objects');g=CrystalLattice()
        g.connect('apple','orange',40,EDGE_ANALOG)
        for label in ('apple','orange','tree'):g.get_or_create(label)
        for mode in ('off','diagnostic','hypotheses'):
            g.configure_geometry(space,'objects',mode)
            for label in ('apple','orange','tree'):g.bind_semantic_sense(label,ids[label])
            before=[(r.id,list(r.connections.typed_items()),r.state) for r in g.resonators.values()]
            g.find_analogies('apple');g.compare_entities('apple','orange');g.rank_analogy_hypotheses('apple')
            self.assertEqual(before,[(r.id,list(r.connections.typed_items()),r.state) for r in g.resonators.values()])
            comparison=g.geometry_comparison('apple','orange')
            self.assertEqual(comparison is None,mode=='off')
            if comparison:self.assertIsNone(comparison['entailed_relation'])

    def test_multitype_and_geometry_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            space,ids=fixture();space.train('objects');g=CrystalLattice()
            g.configure_geometry(space,'objects','diagnostic')
            g.connect('apple','orange',80,EDGE_ANALOG);g.connect('apple','orange',90,EDGE_SYNTAGM)
            g.set_epoch('chapter');g.get_or_create('apple')
            g.bind_semantic_sense('apple',ids['apple']);g.bind_semantic_sense('orange',ids['orange'])
            full=str(Path(d)/'full.cry');delta=str(Path(d)/'delta.cdt')
            CrystalSnapshot.save(g,full);cache=CrystalSnapshot.generate_state_cache(g)
            g.connect('apple','orange',20,EDGE_PART_OF)
            CrystalSnapshot.save_delta(g,delta,cache)
            h=CrystalLattice();self.assertTrue(CrystalSnapshot.load(h,full))
            a=h.get_or_create('apple');b=h.get_or_create('orange')
            self.assertEqual(len(list(a.connections.typed_items())),2)
            self.assertIsNotNone(h.geometry_comparison('apple','orange')['similarity'])
            self.assertTrue(CrystalSnapshot.load(h,delta,cache))
            self.assertEqual(len(list(h.get_or_create('apple').connections.typed_items())),3)
            self.assertEqual(g.semantic_space.to_dict(),h.semantic_space.to_dict())

    def test_sleep_is_idempotent(self):
        g=CrystalLattice()
        for name in ('a','b','c','d','e','f'):
            for i in range(4):g.connect(name,'root:p'+str(i),90)
        for r in g.resonators.values():r.energy=200
        g.defragment()
        before=(len(g.resonators),sum(len(list(r.connections.typed_items())) for r in g.resonators.values()))
        for _ in range(5):g.defragment()
        after=(len(g.resonators),sum(len(list(r.connections.typed_items())) for r in g.resonators.values()))
        self.assertEqual(before,after)

    def test_useful_hierarchy_and_duplicate_rejection(self):
        g=CrystalLattice()
        ids=[g.get_or_create(x).id for x in ('a','b','c')]
        first=g.propose_structural_abstraction(ids[:2])
        self.assertIsNotNone(first)
        self.assertIsNone(g.propose_structural_abstraction(list(reversed(ids[:2]))))
        second=g.propose_structural_abstraction([first,ids[2]])
        self.assertIsNotNone(second)
        self.assertIsNone(g.propose_structural_abstraction([second,first]))

    def test_incremental_sleep_and_restore(self):
        with tempfile.TemporaryDirectory() as d:
            g=CrystalLattice();filename=str(Path(d)/'state.cry')
            for stage in range(4):
                reason=CrystalReason('observation',source_label=f'source{stage}',metadata={'status':'confirmed'})
                g.connect('object'+str(stage),'class',60,EDGE_IS_A,reason)
                for r in g.resonators.values():r.energy=200
                g.defragment();CrystalSnapshot.save(g,filename)
                h=CrystalLattice();CrystalSnapshot.load(h,filename);g=h
                count=len(g.resonators)
                for _ in range(4):g.defragment()
                self.assertEqual(count,len(g.resonators))
                self.assertEqual(len(g.label_to_id),len(g.resonators))
                for node,members in g.abstraction_components.items():
                    self.assertNotIn(node,g._abstraction_leaves(node))
                self.assertFalse(any(g.resonators[node].belief_status=='confirmed' for node in g.abstraction_components))

    def test_geometry_never_changes_energy_algorithm(self):
        import random
        states=[]
        for mode in ('off','diagnostic','hypotheses'):
            space,ids=fixture();space.train('objects');g=CrystalLattice()
            g.connect('apple','orange',60)
            g.get_or_create('apple').energy=1000
            g.configure_geometry(space,'objects',mode)
            for label in ('apple','orange'):g.bind_semantic_sense(label,ids[label])
            random.seed(17)
            g.tick()
            states.append([(r.label,r.energy,int(r.state),r.hdc_vector) for r in g.resonators.values()])
        self.assertEqual(states[0],states[1]);self.assertEqual(states[1],states[2])

    def test_graph_import_requires_confirmed_provenance(self):
        space,ids=fixture();g=CrystalLattice();g.configure_geometry(space,'objects','diagnostic')
        for label in ('apple','fruit'):g.bind_semantic_sense(label,ids[label])
        g.connect('apple','fruit',60,EDGE_IS_A,CrystalReason('input',source_type='text'))
        self.assertEqual(g.learn_semantic_evidence(),0)
        g.connect('apple','fruit',60,EDGE_IS_A,CrystalReason('observation',source_label='experiment',metadata={'status':'confirmed'}))
        self.assertEqual(g.learn_semantic_evidence(),1)
        self.assertEqual(g.learn_semantic_evidence(),0)
