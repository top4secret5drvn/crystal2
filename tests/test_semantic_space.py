import json,math,unittest
from semantic_space import SemanticSpace, Evidence


def fixture(dim=3, seed=17):
    s=SemanticSpace(dim,seed)
    s.add_world('objects',('entity','class'),'synthetic fixture')
    s.add_world('operators',('connector','function'),'synthetic fixture')
    ids={}
    for word in ('apple','orange','tree','grass'):
        ids[word]=s.add_sense(word,word,'entity',['objects'])
    for word in ('fruit','plant'):
        ids[word]=s.add_sense(word,word,'class',['objects'])
    for word,category in [('apple','fruit'),('orange','fruit'),('tree','plant'),('grass','plant')]:
        s.observe(Evidence(word,ids[word],ids[category],'IS_A','objects','fixture',confirmed=True))
    return s,ids

class Geometry(unittest.TestCase):
    def test_dimensions_and_reproducibility(self):
        for dim in (2,3,8):
            a,ids=fixture(dim);b,_=fixture(dim)
            a.train('objects');b.train('objects')
            self.assertEqual(a.to_dict(),b.to_dict())
            self.assertEqual(len(a.position(ids['apple'],'objects')),dim)
            self.assertGreater(a.similarity(ids['apple'],ids['orange'],'objects'),.2)
            self.assertNotEqual(ids['apple'],ids['orange'])
            self.assertEqual(a.train('objects'),0)

    def test_unknown_world_roles_and_noise(self):
        s,ids=fixture();s.train('objects')
        self.assertIsNone(s.similarity(ids['apple'],ids['orange'],'operators'))
        new=s.add_sense('unknown','new','entity',['objects'])
        self.assertIsNone(s.position(new,'objects'))
        with self.assertRaises(ValueError):
            s.observe(Evidence('syn',new,ids['apple'],'SYN','objects','text',confirmed=True))
        self.assertIsNone(s.similarity(ids['apple'],ids['fruit'],'objects'))

    def test_polysemy_and_operator_roles(self):
        s,ids=fixture()
        keys=[s.add_sense('ключ',meaning,'entity',['objects']) for meaning in ('tool','spring','access')]
        self.assertEqual(len(s.resolve('ключ')),3)
        features=[s.add_sense('feature'+str(i),'feature'+str(i),'class',['objects']) for i in range(3)]
        for i,k in enumerate(keys):
            s.observe(Evidence('key'+str(i),k,features[i],'PROPERTY','objects','context',confirmed=True))
        self.assertEqual(s.resolve('ключ','objects',[('PROPERTY',features[0])]),[keys[0]])
        s.add_world('language',('connector','function'),'functional test')
        a=s.add_sense('однако','contrast','connector',['language'])
        b=s.add_sense('потому что','cause','connector',['language'])
        link=s.add_sense('связь','link','function',['language'])
        for x in (a,b):s.observe(Evidence(x,x,link,'ROLE','language','functional annotation',confirmed=True))
        for x,kind in ((a,'opposition'),(b,'explanation')):
            f=s.add_sense(kind,kind,'function',['language'])
            s.observe(Evidence('function-'+x,x,f,'ROLE','language','functional annotation',confirmed=True))
        self.assertNotEqual(s.features(a,'language'),s.features(b,'language'))
        s.train('language')
        self.assertGreater(s.similarity(a,b,'language'),0)
        self.assertNotEqual(a,b)
        self.assertIsNone(s.similarity(a,b,'objects'))
        s.train('objects')
        self.assertIsNone(s.similarity(keys[1],keys[2],'objects')) # spelling creates no constraint

    def test_retraction_conflict_and_unconfirmed(self):
        s,ids=fixture();s.train('objects')
        s.observe(Evidence('no',ids['apple'],ids['fruit'],'IS_A','objects','counterexample',confirmed=True,positive=False))
        s.train('objects')
        self.assertIsNone(s.position(ids['apple'],'objects'))
        s.retract('no');s.train('objects')
        self.assertIsNotNone(s.position(ids['apple'],'objects'))
        s.observe(Evidence('incompatible',ids['apple'],ids['orange'],'INCOMPATIBLE','objects','experiment',confirmed=True))
        s.train('objects')
        self.assertTrue(s.conflicts)
        self.assertIsNone(s.position(ids['apple'],'objects'))

    def test_budget_and_evidence_identity(self):
        s,ids=fixture()
        self.assertLessEqual(s.train('objects',max_operations=9),9)
        e=s.evidence['apple'];version=s.version
        self.assertFalse(s.observe(e));self.assertEqual(s.version,version)
        with self.assertRaises(ValueError):s.train('objects',learning_rate=float('nan'))
        self.assertTrue(all(math.isfinite(v) for p in s.positions.values() for v in p.coordinates))
        restored=SemanticSpace.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(s.to_dict(),restored.to_dict())
        self.assertFalse(s.communities('objects'))
        self.assertTrue(s.communities('objects',min_sources=1))

    def test_sense_multiple_worlds_and_no_budget(self):
        s,ids=fixture()
        s.add_world('applications',('entity','class'),'functional use case')
        sid=s.add_sense('ключ','access','entity',['objects','applications'])
        self.assertIn(sid,s.resolve('ключ','objects'))
        self.assertIn(sid,s.resolve('ключ','applications'))
        self.assertEqual(s.train('objects',max_operations=0),0)
        self.assertIsNone(s.position(ids['apple'],'objects'))

    def test_russian_object_examples(self):
        s=SemanticSpace(3,17);s.add_world('objects',('entity','class'),'example A')
        ids={word:s.add_sense(word,word,role,['objects']) for word,role in
             [('яблоко','entity'),('апельсин','entity'),('дерево','entity'),('фрукт','class'),('растение','class')]}
        for word,category in [('яблоко','фрукт'),('апельсин','фрукт'),('дерево','растение')]:
            s.observe(Evidence(word,ids[word],ids[category],'IS_A','objects','observation',confirmed=True))
        s.train('objects')
        self.assertGreater(s.similarity(ids['яблоко'],ids['апельсин'],'objects'),0)
        self.assertNotEqual(ids['яблоко'],ids['апельсин'])
        self.assertIsNone(s.position(ids['дерево'],'objects'))
