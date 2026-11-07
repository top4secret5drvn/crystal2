"""Experimental contextual geometry; evidence and coordinates never assert facts.

Only explicitly confirmed typed evidence contributes constraints. This module
has no dependency on the lattice, network, or text parser.
"""
from dataclasses import dataclass, asdict
from itertools import combinations
import hashlib
import json
import math
import random


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


@dataclass
class Lexeme:
    id: str
    normalized: str
    forms: tuple
    senses: tuple = ()


@dataclass
class Sense:
    id: str
    lexeme_id: str
    meaning_key: str
    role: str
    worlds: tuple


@dataclass
class SemanticWorld:
    id: str
    roles: tuple
    provenance: str


@dataclass
class ConceptPosition:
    sense_id: str
    world_id: str
    coordinates: tuple
    evidence_count: int
    quality: float
    version: int


@dataclass
class Evidence:
    id: str
    subject: str
    target: str
    relation: str
    world: str
    source: str
    confidence: float = 1.0
    confirmed: bool = False
    positive: bool = True
    epoch: str = ''
    dialog: str = ''
    active: bool = True


class SemanticSpace:
    RELATIONS = {'IS_A', 'PART_OF', 'PROPERTY', 'ROLE', 'ANALOG', 'INCOMPATIBLE', 'CAUSE', 'EFFECT'}

    def __init__(self, dimensions=3, seed=1729):
        if dimensions not in (2, 3, 8):
            raise ValueError('Supported dimensions: 2, 3, 8')
        self.dimensions, self.seed = dimensions, seed
        self.lexemes, self.senses, self.worlds = {}, {}, {}
        self.evidence, self.positions = {}, {}
        self.version = 0
        self.trained_versions = {}
        self.last_operations = 0
        self.conflicts = []
        self.constraints_truncated = False

    def add_world(self, name, roles, provenance):
        if not name or not roles or not provenance:
            raise ValueError('Named world, semantic roles and provenance required')
        world = SemanticWorld(name, tuple(sorted(set(roles))), provenance)
        if name in self.worlds and self.worlds[name] != world:
            raise ValueError('World definition is immutable')
        self.worlds[name] = world
        return name

    def add_sense(self, word, meaning_key, role, worlds, forms=()):
        normalized = ' '.join(word.casefold().split())
        if not normalized or not meaning_key:
            raise ValueError('Explicit sense identity required')
        worlds = tuple(sorted(set(worlds)))
        if not worlds or any(w not in self.worlds or role not in self.worlds[w].roles for w in worlds):
            raise ValueError('Sense role incompatible with world')
        lex_id = stable_id('lexeme', normalized)
        lex = self.lexemes.setdefault(lex_id, Lexeme(lex_id, normalized, (normalized,)))
        lex.forms = tuple(sorted(set(lex.forms) | {f.casefold() for f in forms}))
        sid = stable_id('sense', lex_id, meaning_key)
        sense = Sense(sid, lex_id, meaning_key, role, worlds)
        if sid in self.senses and self.senses[sid] != sense:
            raise ValueError('Sense identity is immutable')
        self.senses[sid] = sense
        lex.senses = tuple(sorted(set(lex.senses) | {sid}))
        return sid

    def resolve(self, word, world=None, context_features=()):
        """Returns candidates; ties and missing evidence remain unresolved."""
        candidates = [s for l in self.lexemes.values() if word.casefold() in l.forms
                      for s in l.senses if world is None or world in self.senses[s].worlds]
        if len(candidates) <= 1 or not context_features:
            return sorted(candidates)
        scored = {sid:len(self.features(sid, world) & set(context_features)) for sid in candidates}
        best = max(scored.values(), default=0)
        return sorted(sid for sid in candidates if scored[sid] == best) if best else sorted(candidates)

    def observe(self, evidence):
        e = evidence
        if (not e.id or not e.source or e.relation not in self.RELATIONS or
            not math.isfinite(e.confidence) or not 0 <= e.confidence <= 1):
            raise ValueError('Identified typed evidence and bounded confidence required')
        if e.world not in self.worlds or any(s not in self.senses or e.world not in self.senses[s].worlds
                                            for s in (e.subject, e.target)):
            raise ValueError('Evidence endpoints must belong to its world')
        if e.id in self.evidence:
            if self.evidence[e.id] != e:
                raise ValueError('Evidence ID collision; retract before adding a new observation ID')
            return False
        self.evidence[e.id] = Evidence(**asdict(e))
        self.version += 1
        return True

    def retract(self, evidence_id):
        e = self.evidence[evidence_id]
        if e.active:
            e.active = False
            self.version += 1

    def _facts(self, world):
        grouped = {}
        for e in self.evidence.values():
            if e.world == world and e.active and e.confirmed and e.confidence > 0:
                grouped.setdefault((e.subject, e.relation, e.target), []).append(e)
        facts = {}
        for key, group in grouped.items():
            strength = {sign:max((e.confidence for e in group if e.positive == sign), default=0)
                        for sign in (False, True)}
            if strength[True] > strength[False]:
                facts[key] = [e for e in group if e.positive]
        return facts

    def features(self, sid, world):
        worlds = [world] if world else self.senses[sid].worlds
        return {(relation, target) for w in worlds for (s, relation, target) in self._facts(w)
                if s == sid and relation not in ('ANALOG', 'INCOMPATIBLE')}

    def constraints(self, world):
        if world not in self.worlds:
            raise ValueError('Unknown world')
        facts = self._facts(world)
        features, pairs = {}, {}
        for (s, relation, t), evidence in sorted(facts.items()):
            if relation not in ('ANALOG','INCOMPATIBLE'):
                features.setdefault((relation,t,self.senses[s].role), []).append((s,evidence))
            elif relation == 'INCOMPATIBLE' or len({e.source for e in evidence}) >= 2:
                pairs.setdefault(tuple(sorted((s,t))), []).append((2.0 if relation=='INCOMPATIBLE' else .3, evidence))
        attempts = 0
        self.constraints_truncated = False
        for _, subjects in sorted(features.items()):
            for (a,ea),(b,eb) in combinations(sorted(subjects), 2):
                if attempts >= 4096:
                    self.constraints_truncated = True
                    break
                attempts += 1
                if a != b:
                    pairs.setdefault(tuple(sorted((a,b))), []).append((.3, ea+eb))
        result=[]; self.conflicts=[]
        for (a,b), signals in sorted(pairs.items()):
            if self.senses[a].role != self.senses[b].role:
                continue
            targets={distance for distance,_ in signals}
            if len(targets)>1:
                self.conflicts.append((a,b,world))
                continue
            ids={e.id for _,es in signals for e in es}
            quality=sum(self.evidence[i].confidence for i in ids)/len(ids)
            result.append((a,b,next(iter(targets)),quality,len(ids)))
        return result

    def train(self, world, iterations=120, max_operations=100000, learning_rate=.04):
        if (not isinstance(iterations,int) or not isinstance(max_operations,int) or
            iterations < 0 or max_operations < 0 or not math.isfinite(learning_rate) or not 0<learning_rate<=.2):
            raise ValueError('Invalid optimization budget')
        if iterations == 0 or max_operations < self.dimensions:
            self.last_operations = 0
            return 0
        constraints=self.constraints(world)
        if self.trained_versions.get(world) == self.version:
            self.last_operations=0
            return 0
        ids=sorted({s for a,b,*_ in constraints for s in (a,b)})
        # Remove unsupported positions after retraction/conflict; unknown != false.
        for key in list(self.positions):
            if key[1]==world and key[0] not in ids:
                del self.positions[key]
        coords={}
        for sid in ids:
            rng=random.Random(stable_id(self.seed,world,sid))
            old=self.positions.get((sid,world))
            coords[sid]=list(old.coordinates if old else [rng.uniform(-1,1) for _ in range(self.dimensions)])
        operations=0
        for _ in range(iterations):
            for a,b,target,quality,count in constraints:
                if operations+self.dimensions>max_operations:
                    break
                delta=[coords[a][i]-coords[b][i] for i in range(self.dimensions)]
                distance=math.sqrt(sum(v*v for v in delta))
                if distance<1e-12:
                    delta=[1.0]+[0.0]*(self.dimensions-1);distance=1.0
                step=learning_rate*quality*(distance-target)/max(distance,1e-12)
                for i in range(self.dimensions):
                    change=max(-.1,min(.1,step*delta[i]))
                    coords[a][i]=max(-10.,min(10.,coords[a][i]-change))
                    coords[b][i]=max(-10.,min(10.,coords[b][i]+change))
                operations+=self.dimensions
            else:
                continue
            break
        for sid in ids:
            related=[c for c in constraints if sid in c[:2]]
            self.positions[sid,world]=ConceptPosition(sid,world,tuple(coords[sid]),
                sum(c[4] for c in related),sum(c[3] for c in related)/len(related),self.version)
        # Budget exhaustion is reported, not silently called convergence.
        if operations < iterations*len(constraints)*self.dimensions:
            self.trained_versions.pop(world,None)
        else:
            self.trained_versions[world]=self.version
        self.last_operations=operations
        return operations

    def position(self, sid, world):
        p=self.positions.get((sid,world))
        return p.coordinates if p and p.version==self.version else None

    def similarity(self, a, b, world):
        if a not in self.senses or b not in self.senses or world not in self.worlds:
            return None
        if any(world not in self.senses[s].worlds for s in (a,b)):
            return None
        if self.senses[a].role != self.senses[b].role:
            return None
        pa,pb=self.positions.get((a,world)),self.positions.get((b,world))
        if not pa or not pb or pa.version!=self.version or pb.version!=self.version:
            return None
        distance=math.dist(pa.coordinates,pb.coordinates)
        quality=min(pa.quality,pb.quality)
        volume=min(1.,min(pa.evidence_count,pb.evidence_count)/4)
        return math.exp(-distance)*quality*volume

    def communities(self, world, min_size=3, min_sources=2):
        """Proposals only: connected components of verified typed facts, never SYN."""
        adjacency={}
        for (a,relation,b), evidence in self._facts(world).items():
            if relation in ('IS_A','PART_OF','PROPERTY','ROLE') and len({e.source for e in evidence}) >= min_sources:
                adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
        visited=set();groups=[]
        for start in sorted(adjacency):
            if start in visited:continue
            pending=[start];component=set()
            while pending:
                sid=pending.pop()
                if sid in component:continue
                component.add(sid);pending.extend(adjacency[sid]-component)
            visited.update(component)
            if len(component)>=min_size:groups.append(tuple(sorted(component)))
        return groups

    def to_dict(self):
        return {'schema':1,'dimensions':self.dimensions,'seed':self.seed,'version':self.version,
                **{name:[asdict(v) for v in getattr(self,name).values()] for name in
                   ('lexemes','senses','worlds','evidence','positions')},
                'trained_versions':self.trained_versions}

    @classmethod
    def from_dict(cls, data):
        if data['schema']!=1:raise ValueError('Unsupported semantic schema')
        space=cls(data['dimensions'],data['seed'])
        space.version=data['version'];space.trained_versions=dict(data['trained_versions'])
        for name,kind in [('lexemes',Lexeme),('senses',Sense),('worlds',SemanticWorld),('evidence',Evidence),('positions',ConceptPosition)]:
            for raw in data[name]:
                obj=kind(**raw)
                for attr in ('forms','senses','worlds','roles','coordinates'):
                    if hasattr(obj,attr):setattr(obj,attr,tuple(getattr(obj,attr)))
                key=(obj.sense_id,obj.world_id) if name=='positions' else obj.id
                getattr(space,name)[key]=obj
        return space
