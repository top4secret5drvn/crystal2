import unittest
from types import SimpleNamespace
from cognitive_modules import ConceptFormationEngine, Reason

class Concepts(unittest.TestCase):
    def test_negative_examples_cost_bits(self):
        kb=SimpleNamespace(facts={},rules=[])
        for i in range(12):
            for p in ('a','b','c'):
                kb.facts[f'{p}(x{i})']=(1 if p!='c' or i<6 else -1,Reason('input'))
        e=ConceptFormationEngine(kb);e.analyze_facts(min_confidence=.5)
        self.assertFalse(e.discovered_concepts)

    def test_repeat_analysis_deduplicates(self):
        kb=SimpleNamespace(facts={},rules=[])
        for i in range(12):
            for p in ('a','b','c'): kb.facts[f'{p}(x{i})']=(1,Reason('input'))
        e=ConceptFormationEngine(kb); e.analyze_facts(); n=len(e.discovered_concepts)
        e.analyze_facts(); self.assertEqual(n,len(e.discovered_concepts))
        e.auto_generalize(); n=len(kb.rules); e.auto_generalize(); self.assertEqual(n,len(kb.rules))
