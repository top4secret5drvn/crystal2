"""
engine.py — Кремниевая подложка Кристалла (v4.0 Cognitive Rewrite).
Объединяет Векторы 1-5, Геном, Причинный Калькулус, Планировщик и КОГНИТИВНЫЙ КОНВЕЙЕР.
"""
import hashlib
import random
from enum import IntEnum, auto
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple, Set, Any

# ============================================================
# 🧩 Раздел 27.1: Битовые тензоры связей (Causal Typology)
# ============================================================
EDGE_SYNTAGM = 0x00000000
EDGE_CAUSE   = 0x10000000
EDGE_EFFECT  = 0x20000000
EDGE_COND    = 0x30000000
EDGE_EXCEPT  = 0x40000000
EDGE_ANALOG  = 0x50000000
EDGE_GOAL    = 0x80000000
EDGE_IS_A    = 0x60000000
EDGE_PART_OF = 0x70000000
MASK_TYPE   = 0xF0000000
MASK_WEIGHT = 0x0FFFFFFF

EDGE_NAMES = {
    EDGE_SYNTAGM: "SYN", EDGE_CAUSE: "CAUSE", EDGE_EFFECT: "EFFECT",
    EDGE_COND: "COND", EDGE_EXCEPT: "EXCEPT", EDGE_ANALOG: "ANALOG",
    EDGE_GOAL: "GOAL", EDGE_IS_A: "IS_A", EDGE_PART_OF: "PART_OF",
}

# 🆕 Приоритет 1.6: Ограничения типов связей
EDGE_CONSTRAINTS = {
    EDGE_SYNTAGM: {"description": "Соседство в тексте", "bidirectional": True},
    EDGE_CAUSE:   {"description": "Причина → Следствие", "bidirectional": False, "reverse": EDGE_EFFECT},
    EDGE_EFFECT:  {"description": "Следствие ← Причина", "bidirectional": False, "reverse": EDGE_CAUSE},
    EDGE_COND:    {"description": "Условие", "bidirectional": True},
    EDGE_EXCEPT:  {"description": "Исключение / Опровержение", "bidirectional": True},
    EDGE_ANALOG:  {"description": "Аналогия", "bidirectional": True},
    EDGE_GOAL:    {"description": "Цель", "bidirectional": False},
    EDGE_IS_A:    {"description": "Классовое включение", "bidirectional": False},
    EDGE_PART_OF: {"description": "Часть целого", "bidirectional": False},
}

def pack_edge(weight: int, edge_type: int = EDGE_SYNTAGM) -> int:
    """Упаковывает вес и тип в 32-битное слово за 1 такт."""
    return (min(weight, MASK_WEIGHT) & MASK_WEIGHT) | (edge_type & MASK_TYPE)

def unpack_edge(packed: int) -> Tuple[int, int]:
    """Извлекает (вес, тип) из 32-битного слова за 1 такт."""
    return packed & MASK_WEIGHT, packed & MASK_TYPE

def edge_type_name(packed: int) -> str:
    """Человекочитаемое имя типа связи."""
    _, etype = unpack_edge(packed)
    return EDGE_NAMES.get(etype, f"UNK({etype:#x})")

# ============================================================
# Базовые структуры
# ============================================================
class TruthValue(IntEnum):
    VOID = 0b00
    TRUE = 0b01
    FALSE = 0b10
    PARADOX = 0b11

# 🆕 Приоритет 1.1: CrystalReason - перенос идеи Reason из Вердикта
@dataclass
class CrystalReason:
    """Причина убеждения или связи (адаптация Reason из Вердикта)."""
    kind: str                    # "input", "association", "insight", "sleep", "rule", "hypothesis", "conflict", "defeater"
    source_label: str = ""       # метка узла или источника
    source_type: str = ""        # "user", "text", "wiki", "sleep", "marker_collision", "analogy"
    parents: List['CrystalReason'] = field(default_factory=list)
    confidence: float = 1.0
    context: str = "global"      # имя эпохи или контекста
    timestamp: int = 0           # такт создания
    metadata: dict = field(default_factory=dict)
    
    def chain_length(self) -> int:
        """Длина цепочки причин (для Бритвы Оккама)."""
        nodes = set()
        self._collect_nodes(nodes)
        return len(nodes)
    
    def _collect_nodes(self, nodes: set):
        """Рекурсивный сбор всех узлов в цепочке."""
        if id(self) in nodes: return
        nodes.add(id(self))
        for p in self.parents:
            p._collect_nodes(nodes)
    
    def collect_sources(self) -> List[str]:
        """Собрать все листовые источники."""
        sources = []
        self._collect_leaves(sources, set())
        return sources
    
    def _collect_leaves(self, sources: list, seen: set):
        """Рекурсивный сбор листьев."""
        if id(self) in seen: return
        seen.add(id(self))
        if not self.parents:
            if self.kind in ("input", "wiki", "text"):
                sources.append(f"{self.source_type}:{self.source_label}")
        else:
            for p in self.parents:
                p._collect_leaves(sources, seen)
    
    def generate_report(self) -> str:
        """Текстовый XAI-отчёт (по аналогии с Вердиктом)."""
        sources = self.collect_sources()
        complexity = self.chain_length()
        report = f"=== XAI ОТЧЕТ ===\n"
        report += f"Тип: {self.kind} ({self.source_type})\n"
        report += f"Источник: {self.source_label}\n"
        report += f"Уверенность: {self.confidence * 100:.0f}%\n"
        report += f"Контекст: {self.context}\n"
        report += f"Сложность: {complexity} шагов\n"
        if complexity > 3:
            report += f"⚠️ Переусложненная цепочка! (Применен штраф Оккама)\n"
        if sources:
            report += f"Источники: {', '.join(set(sources))}\n"
        report += f"=================\n"
        return report
    
    def __str__(self):
        return f"{self.kind}({self.source_label}) [{self.confidence:.2f}]"

# 🆕 Приоритет 1.4: Контексты
@dataclass
class ContextNode:
    """Узел иерархии контекстов."""
    name: str
    parent: Optional[str] = None
    context_type: str = "epoch"  # "global", "epoch", "hypothesis", "dialog", "wiki"
    created_tick: int = 0

class HDCEncoder:
    DIM = 10000
    BYTES_PER_VECTOR = (DIM + 7) // 8
    
    def __init__(self):
        self._cache: Dict[bytes, int] = {}
    
    def encode(self, data: str | bytes) -> int:
        if isinstance(data, str):
            data = data.encode('utf-8')
        if data in self._cache:
            return self._cache[data]
        
        vector_bytes = bytearray(self.BYTES_PER_VECTOR)
        seed = hashlib.sha256(data).digest()
        pos, iteration = 0, 0
        
        while pos < self.BYTES_PER_VECTOR:
            block_input = seed + iteration.to_bytes(4, 'little')
            block = hashlib.sha256(block_input).digest()
            chunk = min(32, self.BYTES_PER_VECTOR - pos)
            vector_bytes[pos:pos + chunk] = block[:chunk]
            pos += chunk
            iteration += 1
        
        if self.DIM % 8 != 0:
            mask = (1 << (self.DIM % 8)) - 1
            vector_bytes[-1] &= mask
        
        vector_int = int.from_bytes(vector_bytes, 'little')
        if len(self._cache) < 10000:
            self._cache[data] = vector_int
        return vector_int
    
    def bind(self, v1: int, v2: int) -> int:
        """Связывание двух векторов (Binding / XOR)."""
        return v1 ^ v2
    
    def bundle(self, vectors: List[int]) -> int:
        if not vectors:
            return 0
        if len(vectors) == 1:
            return vectors[0]
        
        result_bytes = bytearray(self.BYTES_PER_VECTOR)
        threshold = len(vectors) / 2
        vec_bytes = [v.to_bytes(self.BYTES_PER_VECTOR, 'little') for v in vectors]
        
        for byte_idx in range(self.BYTES_PER_VECTOR):
            bit_counts = [0] * 8
            for vb in vec_bytes:
                byte_val = vb[byte_idx]
                for i in range(8):
                    if byte_val & (1 << i):
                        bit_counts[i] += 1
            res_byte = 0
            for i in range(8):
                if bit_counts[i] > threshold:
                    res_byte |= (1 << i)
            result_bytes[byte_idx] = res_byte
        
        return int.from_bytes(result_bytes, 'little')
    
    def similarity(self, v1: int, v2: int) -> float:
        diff = v1 ^ v2
        diff_bits = diff.bit_count() if hasattr(diff, 'bit_count') else bin(diff).count('1')
        return 1.0 - (diff_bits / self.DIM)
    
    def generate_random(self) -> int:
        """Генерирует случайный ортогональный гипервектор."""
        return random.getrandbits(self.DIM)
    
    def invert(self, vector: int) -> int:
        """Побитовая инверсия вектора (антипод -P)."""
        mask = (1 << self.DIM) - 1
        return (~vector) & mask
    
    def bundle_pair(self, vector_a: int, vector_b: int) -> int:
        """Нахождение общего базиса двух векторов (Структуралистский Bundle)."""
        mask = (1 << self.DIM) - 1
        match_mask = (~(vector_a ^ vector_b)) & mask
        diff_mask = (vector_a ^ vector_b) & mask
        kept_bits = vector_a & match_mask
        random_fill = self.generate_random()
        random_bits = random_fill & diff_mask
        return (kept_bits | random_bits) & mask

# ============================================================
# 🧠 Раздел 40: Когнитивные структуры (Symbolic AI / 80s)
# ============================================================
class MarkerType(IntEnum):
    ACTIVATION = auto()
    SEEKER = auto()
    PLANNER = auto()
    CRITIC = auto()
    BIND = auto()

@dataclass
class Marker:
    type: MarkerType
    origin_id: int
    energy: int
    payload: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    color: int = 0

@dataclass
class FrameSlot:
    role: str
    filler_id: Optional[int] = None
    activation: float = 0.0

@dataclass
class CognitiveFrame:
    frame_id: int
    label: str
    slots: Dict[str, FrameSlot] = field(default_factory=dict)
    activation: float = 0.0
    evoking_node_id: Optional[int] = None
    
    def bind_slot(self, role: str, filler_id: int, activation: float):
        if role not in self.slots:
            self.slots[role] = FrameSlot(role)
        if activation >= self.slots[role].activation:
            self.slots[role].filler_id = filler_id
            self.slots[role].activation = activation
    
    def get_filled_roles(self) -> Dict[str, int]:
        """Возвращает словарь заполненных ролей: {роль: id_узла}."""
        return {role: slot.filler_id for role, slot in self.slots.items() if slot.filler_id is not None}

@dataclass
class DependencyNode:
    """Узел дерева зависимостей для генерации речи."""
    node_id: int
    role: str
    children: List['DependencyNode'] = field(default_factory=list)
    
    def linearize(self, lattice) -> List[str]:
        """In-order обход для русского языка (SVO: Субъект -> Глагол -> Объект)."""
        words = []
        for child in self.children:
            if child.role in ("SUBJ", "CAUSE", "ENTITY"):
                words.extend(child.linearize(lattice))
        if self.node_id in lattice.resonators:
            lbl = lattice.resonators[self.node_id].label
            clean = lbl[5:] if lbl.startswith("root:") else lbl
            if clean and not clean.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')) and clean != 'SELF':
                if clean not in words:
                    words.append(clean)
        for child in self.children:
            if child.role in ("OBJ", "EFFECT", "PROP", "GOAL"):
                words.extend(child.linearize(lattice))
        return words

@dataclass
class Construction:
    """Construction Grammar: Связь семантического паттерна и синтаксиса."""
    name: str
    meaning_pattern: Dict[str, str]
    syntactic_template: List[str]
    weight: float = 1.0

@dataclass
class Hypothesis:
    module: str
    content: Any
    confidence: float
    timestamp: int = 0

class CognitiveBlackboard:
    """Доска объявлений для модулей (BlackBoard Architecture)."""
    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self.conceptual_plan: List[int] = []
        self.intention_vector: Optional[int] = None
    
    def post_hypothesis(self, module: str, content: Any, confidence: float, tick: int):
        self.hypotheses.append(Hypothesis(module, content, confidence, timestamp=tick))
    
    def clear_hypotheses(self):
        self.hypotheses.clear()

# ============================================================
# Резонатор (Узел решетки)
# ============================================================
@dataclass
class Resonator:
    id: int
    label: str
    hdc_vector: int
    state: TruthValue = TruthValue.VOID
    energy: int = 0
    connections: Dict[int, int] = field(default_factory=dict)
    defeats: List[int] = field(default_factory=list)
    last_tick: int = 0
    activation_sources: List[int] = field(default_factory=list)
    context_sources: List[int] = field(default_factory=list)
    context_mask: int = 0
    cognitive_role: Optional[str] = None
    
    # 🆕 Приоритет 1.3: Статусы убеждений
    belief_status: str = "unknown"  # "unknown", "observed", "confirmed", "hypothesis", "defeated", "contradiction"
    belief_confidence: float = 0.0
    belief_reason: Optional[CrystalReason] = None
    belief_context: str = "global"
    
    def inject_energy(self, amount: int, tick: int, source_ids: List[int] = None, cap: int = 5000):
        self.energy += amount
        if self.energy > cap:
            self.energy = cap
        self.last_tick = tick
        if source_ids:
            for src_id in source_ids:
                if src_id not in self.activation_sources:
                    self.activation_sources.append(src_id)
                    if len(self.activation_sources) > 5:
                        self.activation_sources.pop(0)
    
    def decay(self, base_decay: int = 25):
        """🧬 Затухание управляется Геномом. Нелинейное: чем выше энергия, тем быстрее падает."""
        if self.energy < 20:
            self.energy = 0
        else:
            if self.energy > 2000:
                factor = min(90, base_decay * 4)
            elif self.energy > 500:
                factor = min(70, base_decay * 2)
            else:
                factor = base_decay
            self.energy = (self.energy * (100 - factor)) // 100
    
    def is_active(self, threshold: int = 15) -> bool:
        return self.energy >= threshold
    
    def add_context(self, hdc: int):
        if len(self.context_sources) < 50:
            self.context_sources.append(hdc)
    
    def get_connections_by_type(self, edge_type: int) -> List[Tuple[int, int]]:
        result = []
        for target_id, packed in self.connections.items():
            w, et = unpack_edge(packed)
            if et == edge_type:
                result.append((target_id, w))
        return result
    
    def has_incoming_type(self, lattice: 'CrystalLattice', edge_type: int) -> bool:
        for r in lattice.resonators.values():
            if self.id in r.connections:
                _, et = unpack_edge(r.connections[self.id])
                if et == edge_type:
                    return True
        return False

# ============================================================
# 🧬 Когнитивный Геном (Раздел 23)
# ============================================================
GENE_DECAY_SHIFT      = 0
GENE_ENTROPY_THRESH   = 32
GENE_MUTATION_RATE    = 64
GENE_MAX_DEPTH        = 128
GENE_ANALOGY_TOL      = 160
GENE_PARADOX_PENALTY  = 192
GENE_PHASE_LOCK       = 224

# ============================================================
# CrystalLattice (Ядро)
# ============================================================
class CrystalLattice:
    def __init__(self, max_resonators: int = 100000):
        self.encoder = HDCEncoder()
        from calibration import CalibrationProfile
        self.calibration = CalibrationProfile()
        self.resonators: Dict[int, Resonator] = {}
        self.label_to_id: Dict[str, int] = {}
        self.tick_count = 0
        self._next_id = 0
        self.interference_log: List[str] = []
        self.paradox_log: List[str] = []
        self.false_antagonisms: Set[Tuple[str, str]] = set()
        self.is_dormant = False
        self.dormancy_friction = 0.1
        self._last_read_count = 0
        
        # 🧬 Когнитивный Геном
        self.genome: int = (
            (25 << GENE_DECAY_SHIFT) |
            (800 << GENE_ENTROPY_THRESH) |
            (150 << GENE_MUTATION_RATE) |
            (10 << GENE_MAX_DEPTH) |
            (500 << GENE_ANALOGY_TOL) |
            (100 << GENE_PARADOX_PENALTY) |
            (1 << GENE_PHASE_LOCK)
        )
        
        # 🚀 Вектор 5
        self.self_id = self._create_self_resonator()
        self.epochs: Dict[str, int] = {}
        self.skills: List[int] = []
        self.current_epoch: Optional[str] = None
        self.last_query_path: List[int] = []
        self.last_query_label: Optional[str] = None
        
        # 🧠 Раздел 40: Когнитивный конвейер
        self.marker_queue: List[Marker] = []
        self.active_frames: Dict[int, CognitiveFrame] = {}
        self.constructions_db: List[Construction] = []
        self._next_frame_id = 0
        self.blackboard = CognitiveBlackboard()
        self._register_primitive_frames()
        
        # 🆕 Приоритет 1.2: Метаданные связей
        self.edge_meta: Dict[Tuple[int, int], CrystalReason] = {}
        
        # 🆕 Приоритет 1.4: Контексты
        self.contexts: Dict[str, ContextNode] = {"global": ContextNode("global")}
        self.active_context: str = "global"
        
        # 🆕 Приоритет 1.5: Гипотезы
        self.hypotheses: Dict[int, CrystalReason] = {}
        
        # 🆕 Приоритет 1.11: Надёжность источников
        self.source_reliability: Dict[str, float] = {
            "user": 1.0,
            "text_file": 0.8,
            "wiki": 0.6,
            "sleep_abstraction": 0.4,
            "marker_collision": 0.3,
            "analogy": 0.3,
            "hypothesis": 0.2,
        }
    
    def _register_primitive_frames(self):
        """Регистрация базовых фреймов (Fillmore)."""
        primitives = [
            ("EVENT", ["Agent", "Action", "Patient"]),
            ("MOTION", ["Mover", "Path", "Goal"]),
            ("COMMERCE", ["Buyer", "Seller", "Goods", "Money"]),
            ("COMMUNICATION", ["Speaker", "Hearer", "Message"]),
            ("CAUSATION", ["Cause", "Effect"]),
            ("STATE", ["Entity", "Property"]),
        ]
        for label, roles in primitives:
            fid = self._next_frame_id
            self._next_frame_id += 1
            frame = CognitiveFrame(frame_id=fid, label=label)
            for role in roles:
                frame.slots[role] = FrameSlot(role)
            self.active_frames[fid] = frame
    
    # --- Геном (свойства) ---
    def _get_gene(self, shift: int, mask: int = 0xFFFFFFFF) -> int:
        return (self.genome >> shift) & mask
    
    @property
    def decay_percent(self) -> int:
        return self._get_gene(GENE_DECAY_SHIFT)
    
    @property
    def entropy_threshold(self) -> float:
        return self._get_gene(GENE_ENTROPY_THRESH) / 1000.0
    
    @property
    def mutation_rate(self) -> float:
        return self._get_gene(GENE_MUTATION_RATE, 0xFFFFFFFFFFFFFFFF) / 10000.0
    
    @property
    def max_depth(self) -> int:
        return max(1, self._get_gene(GENE_MAX_DEPTH))
    
    # --- SELF ---
    def _create_self_resonator(self) -> int:
        hdc = self.encoder.encode("SELF")
        res = Resonator(id=self._next_id, label="SELF", hdc_vector=hdc)
        res.energy = 999999
        res.state = TruthValue.TRUE
        self.resonators[self._next_id] = res
        self.label_to_id["SELF"] = self._next_id
        self._next_id += 1
        return res.id
    
    def set_epoch(self, epoch_name: str):
        if epoch_name not in self.epochs:
            self.epochs[epoch_name] = self.encoder.encode(f"EPOCH:{epoch_name}")
        self.current_epoch = epoch_name
        if not hasattr(self, '_epoch_counters'):
            self._epoch_counters = {}
        if epoch_name not in self._epoch_counters:
            self._epoch_counters[epoch_name] = len(self._epoch_counters) + 1
        self.current_epoch_id = self._epoch_counters[epoch_name]
        
        # 🆕 Приоритет 1.4: Регистрация контекста
        if epoch_name not in self.contexts:
            self.contexts[epoch_name] = ContextNode(
                name=epoch_name,
                parent="global",
                context_type="epoch",
                created_tick=self.tick_count
            )
        self.active_context = epoch_name
    
    def _generate_mutation_mask(self, seed: int, dim: int = 10000, rate: float = 0.015) -> int:
        mask = 0
        state = seed & 0xFFFFFFFF
        count = int(dim * rate)
        flipped = 0
        while flipped < count:
            state = (state * 1103515245 + 12345) & 0xFFFFFFFF
            bit_pos = state % dim
            if not (mask & (1 << bit_pos)):
                mask |= (1 << bit_pos)
                flipped += 1
        return mask
    
    # --- Создание и связывание ---
    def get_or_create(self, label: str) -> Resonator:
        if label not in self.label_to_id:
            new_id = self._next_id
            self._next_id += 1
            hdc = self.encoder.encode(label)
            if getattr(self, 'current_epoch', None) and label != "SELF" and not label.startswith("EPOCH:"):
                hdc ^= self.epochs[self.current_epoch]
            res = Resonator(id=new_id, label=label, hdc_vector=hdc)
            if getattr(self, 'current_epoch_id', 0) != 0 and label != "SELF" and not label.startswith("EPOCH:"):
                res.context_mask = self.current_epoch_id
            self.resonators[new_id] = res
            self.label_to_id[label] = new_id
        return self.resonators[self.label_to_id[label]]
    
    def validate_edge(self, source_id: int, target_id: int, edge_type: int) -> Tuple[bool, str]:
        """🆕 Приоритет 1.6: Валидация создаваемой связи."""
        # Проверка допустимости типа
        if edge_type not in EDGE_CONSTRAINTS:
            return False, f"Неизвестный тип связи: {edge_type:#x}"
        
        # Проверка циклов для причинных связей
        if edge_type in (EDGE_CAUSE, EDGE_EFFECT):
            if source_id in self.resonators and target_id in self.resonators:
                target_r = self.resonators[target_id]
                if source_id in target_r.connections:
                    _, rev_type = unpack_edge(target_r.connections[source_id])
                    if rev_type == EDGE_CAUSE and edge_type == EDGE_CAUSE:
                        return False, f"Цикл причинности: {self.resonators[source_id].label} ↔ {target_r.label}"
        
        # Проверка противоречий с EXCEPT
        # 🛡 ИСПРАВЛЕНО: проверяем и читаем один и тот же ключ (source_id)
        if edge_type != EDGE_EXCEPT:
            if source_id in self.resonators and target_id in self.resonators:
                target_r = self.resonators[target_id]
                if source_id in target_r.connections:
                    _, existing_type = unpack_edge(target_r.connections[source_id])
                    if existing_type == EDGE_EXCEPT:
                        return False, f"Противоречие: уже есть EXCEPT-связь"
        
        return True, "OK"
    
    def connect(self, label1: str, label2: str, weight: int = 10, edge_type: int = EDGE_SYNTAGM, reason: Optional[CrystalReason] = None):
        if getattr(self, 'is_dormant', False):
            weight = int(weight * self.dormancy_friction)
            if weight < 1:
                return
        
        r1 = self.get_or_create(label1)
        r2 = self.get_or_create(label2)
        
        # 🆕 Приоритет 1.6: Валидация
        valid, msg = self.validate_edge(r1.id, r2.id, edge_type)
        if not valid:
            self.paradox_log.append(f"Такт {self.tick_count}: ⚠️ Связь отклонена: {msg}")
            return
        
        packed = pack_edge(weight, edge_type)
        
        if edge_type == EDGE_SYNTAGM:
            old_packed = r1.connections.get(r2.id, 0)
            old_w, _ = unpack_edge(old_packed)
            packed = pack_edge(old_w + weight, edge_type)
            r1.connections[r2.id] = packed
            old_packed2 = r2.connections.get(r1.id, 0)
            old_w2, _ = unpack_edge(old_packed2)
            r2.connections[r1.id] = pack_edge(old_w2 + weight, edge_type)
        else:
            r1.connections[r2.id] = packed
            reverse_type = EDGE_EFFECT if edge_type == EDGE_CAUSE else \
                           EDGE_CAUSE if edge_type == EDGE_EFFECT else edge_type
            r2.connections[r1.id] = pack_edge(weight, reverse_type)
        
        # 🆕 Приоритет 1.2: Сохранение метаданных
        if reason:
            self.edge_meta[(r1.id, r2.id)] = reason
    
    def connect_ids(self, id1: int, id2: int, weight: int = 10, edge_type: int = EDGE_SYNTAGM, reason: Optional[CrystalReason] = None):
        if id1 not in self.resonators or id2 not in self.resonators:
            return
        
        r1, r2 = self.resonators[id1], self.resonators[id2]
        
        # 🆕 Приоритет 1.6: Валидация
        valid, msg = self.validate_edge(id1, id2, edge_type)
        if not valid:
            self.paradox_log.append(f"Такт {self.tick_count}: ⚠️ Связь отклонена: {msg}")
            return
        
        packed = pack_edge(weight, edge_type)
        r1.connections[id2] = packed
        reverse_type = EDGE_EFFECT if edge_type == EDGE_CAUSE else \
                       EDGE_CAUSE if edge_type == EDGE_EFFECT else edge_type
        r2.connections[id1] = pack_edge(weight, reverse_type)
        
        # 🆕 Приоритет 1.2: Сохранение метаданных
        if reason:
            self.edge_meta[(id1, id2)] = reason
    
    # 🆕 Приоритет 1.11: Эффективная уверенность
    def get_effective_confidence(self, reason: CrystalReason) -> float:
        """Учитывает надёжность источника."""
        base = reason.confidence
        source_rel = self.source_reliability.get(reason.source_type, 0.5)
        return base * source_rel
    
    # 🆕 Приоритет 1.3: Статусы убеждений
    def set_belief_status(self, node_id: int, status: str, confidence: float, reason: Optional[CrystalReason] = None, context: str = None):
        """Установить статус убеждения для узла."""
        if node_id not in self.resonators:
            return
        r = self.resonators[node_id]
        r.belief_status = status
        r.belief_confidence = confidence
        if reason:
            r.belief_reason = reason
        if context:
            r.belief_context = context
        else:
            r.belief_context = self.active_context
    
    def get_belief_status(self, node_id: int) -> dict:
        """Получить статус убеждения узла."""
        if node_id not in self.resonators:
            return {}
        r = self.resonators[node_id]
        return {
            "status": r.belief_status,
            "confidence": r.belief_confidence,
            "reason": r.belief_reason,
            "context": r.belief_context
        }
    
    def resolve_belief_conflict(self, node_id: int, new_status: str, new_confidence: float, new_reason: CrystalReason):
        """🆕 Приоритет 1.3: Разрешение конфликта убеждений."""
        if node_id not in self.resonators:
            return
        
        r = self.resonators[node_id]
        old_status = r.belief_status
        old_confidence = r.belief_confidence
        
        # Вычисляем эффективные уверенности
        new_eff = self.get_effective_confidence(new_reason)
        old_eff = old_confidence * self.source_reliability.get(r.belief_reason.source_type if r.belief_reason else "unknown", 0.5)
        
        if new_eff > old_eff:
            # Новый источник надёжнее - принимаем
            r.belief_status = new_status
            r.belief_confidence = new_confidence
            r.belief_reason = new_reason
            r.belief_context = new_reason.context
            
            # Старый статус помечаем как побеждённый
            if old_reason := r.belief_reason:
                self.paradox_log.append(
                    f"Такт {self.tick_count}: 🔄 Ревизия: '{r.label}' {old_status} → {new_status} "
                    f"(уверенность {old_eff:.2f} → {new_eff:.2f})"
                )
        else:
            # Старый надёжнее - создаём конфликт
            r.belief_status = "contradiction"
            self.paradox_log.append(
                f"Такт {self.tick_count}: ⚠️ Конфликт: '{r.label}' не может быть {new_status} "
                f"(старый источник надёжнее: {old_eff:.2f} > {new_eff:.2f})"
            )
    
    # 🆕 Приоритет 1.4: Контексты
    def resolve_context(self, query_context: str = None) -> List[str]:
        """Возвращает цепочку контекстов от текущего к глобальному."""
        ctx = query_context or self.active_context
        chain = [ctx]
        current = ctx
        while current in self.contexts and self.contexts[current].parent:
            parent = self.contexts[current].parent
            if parent in chain: break
            chain.append(parent)
            current = parent
        return chain
    
    # 🆕 Приоритет 1.5: Гипотезы
    def add_hypothesis(self, node_id: int, reason: CrystalReason):
        """Добавить гипотезу (не меняет state)."""
        if node_id not in self.resonators:
            return
        self.hypotheses[node_id] = reason
        self.set_belief_status(node_id, "hypothesis", reason.confidence, reason)
    
    def confirm_hypothesis(self, node_id: int):
        """Подтвердить гипотезу."""
        if node_id not in self.hypotheses:
            return
        reason = self.hypotheses.pop(node_id)
        self.set_belief_status(node_id, "confirmed", reason.confidence, reason)
        # Теперь можно менять state
        if node_id in self.resonators:
            r = self.resonators[node_id]
            if r.state == TruthValue.VOID:
                r.state = TruthValue.TRUE
    
    def reject_hypothesis(self, node_id: int):
        """Отклонить гипотезу."""
        if node_id in self.hypotheses:
            del self.hypotheses[node_id]
        self.set_belief_status(node_id, "defeated", 0.0)
    
    def _apply_structural_antagonism(self, node_a: 'Resonator', node_b: 'Resonator'):
        """
        🧬 Структурализм Соссюра: Математическое разведение векторов.
        """
        basis = self.encoder.bundle_pair(node_a.hdc_vector, node_b.hdc_vector)
        p_axis = self.encoder.generate_random()
        node_a.hdc_vector = self.encoder.bind(basis, p_axis)
        node_b.hdc_vector = self.encoder.bind(basis, self.encoder.invert(p_axis))
        self.connect_ids(node_a.id, node_b.id, weight=1, edge_type=EDGE_EXCEPT)
        self.paradox_log.append(
            f"Такт {self.tick_count}: 🎭 Соссюр: Векторы '{node_a.label}' и '{node_b.label}' "
            f"разведены через инверсию оси P (Топологический парадокс)."
        )
    
    def _scan_for_antonyms_in_sleep(self, similarity_threshold: float = 0.92, min_common_neighbors: int = 5) -> int:
        """
        💤 Фаза сна: Поиск топологических парадоксов.
        """
        candidates = [
            r for r in self.resonators.values()
            if r.label != 'SELF' and
               not r.label.startswith(('EPOCH:', 'mod:', 'mdl:', 'cluster:', 'skill:')) and
               len(r.connections) >= min_common_neighbors
        ]
        antagonisms_found = 0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                r_a = candidates[i]
                r_b = candidates[j]
                sim = self.encoder.similarity(r_a.hdc_vector, r_b.hdc_vector)
                if sim < similarity_threshold:
                    continue
                packed_edge = r_a.connections.get(r_b.id, 0)
                _, edge_type = unpack_edge(packed_edge)
                if edge_type == EDGE_EXCEPT:
                    continue
                if edge_type == EDGE_SYNTAGM:
                    continue
                neighbors_a = {
                    tgt_id for tgt_id, packed in r_a.connections.items()
                    if unpack_edge(packed)[1] == EDGE_SYNTAGM
                }
                neighbors_b = {
                    tgt_id for tgt_id, packed in r_b.connections.items()
                    if unpack_edge(packed)[1] == EDGE_SYNTAGM
                }
                common_neighbors = neighbors_a & neighbors_b
                if len(common_neighbors) >= min_common_neighbors:
                    self._apply_structural_antagonism(r_a, r_b)
                    antagonisms_found += 1
        return antagonisms_found
    
    # --- Defeater ---
    def add_defeater(self, label_defeater: str, label_target: str):
        r_def = self.get_or_create(label_defeater)
        r_tgt = self.get_or_create(label_target)
        if r_tgt.id not in r_def.defeats:
            r_def.defeats.append(r_tgt.id)
        r_def.connections[r_tgt.id] = pack_edge(255, EDGE_EXCEPT)
        
        # 🆕 Приоритет 1.7: Сохраняем причину
        reason = CrystalReason(
            kind="defeater",
            source_label=label_defeater,
            source_type="user",
            confidence=1.0,
            context=self.active_context,
            timestamp=self.tick_count
        )
        self.edge_meta[(r_def.id, r_tgt.id)] = reason
        
        if r_def.is_active() and r_def.state == TruthValue.TRUE:
            self.apply_defeater(r_def.id)
    
    def mark_false_antagonism(self, label1: str, label2: str):
        """🔬 RLHF: Помечает пару как ложный антагонизм."""
        def _clean(label: str) -> str:
            for prefix in ("root:", "cluster:", "mdl:", "skill:", "mod:", "EPOCH:"):
                if label.startswith(prefix):
                    return label[len(prefix):]
            return label
        pair = tuple(sorted((_clean(label1).lower(), _clean(label2).lower())))
        self.false_antagonisms.add(pair)
    
    def _is_false_antagonism(self, l1: str, l2: str) -> bool:
        """Проверяет, помечена ли пара как ложный антагонизм."""
        def _clean(label: str) -> str:
            for prefix in ("root:", "cluster:", "mdl:", "skill:", "mod:", "EPOCH:"):
                if label.startswith(prefix):
                    return label[len(prefix):]
            return label
        pair = tuple(sorted((_clean(l1).lower(), _clean(l2).lower())))
        return pair in self.false_antagonisms
    
    def apply_defeater(self, defeater_id: int):
        defeater = self.resonators.get(defeater_id)
        if not defeater or defeater.state != TruthValue.TRUE:
            return
        
        for target_id in defeater.defeats:
            if target_id in self.resonators:
                target = self.resonators[target_id]
                old_state = target.state
                old_status = target.belief_status
                
                # 🆕 Приоритет 1.7: Усиленное опровержение
                if target_id in self.hypotheses:
                    # Гипотезу просто отклоняем
                    self.reject_hypothesis(target_id)
                    continue
                
                if old_state == TruthValue.TRUE:
                    target.state = TruthValue.PARADOX
                    self.paradox_log.append(f"Такт {self.tick_count}: 🚨 Диссонанс! '{target.label}' TRUE → PARADOX (опровергнут '{defeater.label}')")
                elif old_state == TruthValue.FALSE:
                    target.state = TruthValue.PARADOX
                    self.paradox_log.append(f"Такт {self.tick_count}: 🚨 Диссонанс! '{target.label}' FALSE → PARADOX")
                elif old_state == TruthValue.VOID:
                    target.state = TruthValue.FALSE
                    self.paradox_log.append(f"Такт {self.tick_count}: 🛡 Опровержение: '{target.label}' → FALSE ('{defeater.label}')")
                
                # Сохраняем причину опровержения
                if edge_reason := self.edge_meta.get((defeater.id, target_id)):
                    target.belief_reason = edge_reason
                
                target.belief_status = "defeated"
                
                if target.state in (TruthValue.FALSE, TruthValue.PARADOX):
                    target.energy = 0
    
    # --- Парадоксы ---
    def detect_paradoxes_bitwise(self) -> List[int]:
        if not self.resonators:
            return []
        max_id = max(self.resonators.keys())
        mask_55 = ((1 << ((max_id + 1) * 2)) - 1) // 3
        block = 0
        for r in self.resonators.values():
            block |= (r.state << (r.id * 2))
        even_bits = block & mask_55
        odd_bits_shifted = (block >> 1) & mask_55
        paradox_mask = even_bits & odd_bits_shifted
        paradox_ids = []
        temp_mask = paradox_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            pos = lsb.bit_length() - 1
            target_id = pos // 2
            if target_id in self.resonators:
                paradox_ids.append(target_id)
            temp_mask &= ~(3 << (target_id * 2))
        return paradox_ids
    
    def anneal_paradoxes(self) -> int:
        paradox_ids = self.detect_paradoxes_bitwise()
        if not paradox_ids:
            return 0
        annealed_count = 0
        for pid in paradox_ids:
            r = self.resonators[pid]
            seed = (r.id * 2654435761) ^ self.tick_count
            mutation_mask = self._generate_mutation_mask(seed, self.encoder.DIM, self.mutation_rate)
            r.hdc_vector ^= mutation_mask
            r.state = TruthValue.VOID
            r.energy = 0
            r.context_sources.clear()
            r.context_mask = 0
            self.paradox_log.append(f"Такт {self.tick_count}: 🔥 Отжиг: '{r.label}' мутировал.")
            self.inject_marker(Marker(type=MarkerType.CRITIC, origin_id=pid, energy=50, color=pid))
            annealed_count += 1
        return annealed_count
    
    # --- Инжекция ---
    def inject(self, labels: List[str], energy: int = 100):
        for label in labels:
            r = self.get_or_create(label)
            # 🆕 Приоритет 1.5: Гипотезы учитываются с пониженным весом
            if r.id in self.hypotheses:
                energy = int(energy * 0.3)
            r.inject_energy(energy, self.tick_count, cap=self.calibration.energy_cap)
            self.inject_marker(Marker(
                type=MarkerType.PLANNER,
                origin_id=r.id,
                energy=energy,
                payload={"intent": "EXPLAIN"},
                color=r.id
            ))
    
    # ============================================================
    # 🧠 Раздел 40: Когнитивный Конвейер (Интеграция в Ядро)
    # ============================================================
    def inject_marker(self, marker: Marker):
        """Запуск маркера в сеть (Marker Passing)."""
        self.marker_queue.append(marker)
    
    def tick_markers(self, max_steps: int = 3) -> List[int]:
        """
        Один цикл пробега маркеров.
        """
        if not self.marker_queue:
            return []
        next_queue = []
        collisions = []
        node_markers = {}
        for marker in self.marker_queue:
            if marker.energy <= 0 or marker.step >= max_steps:
                continue
            r = self.resonators.get(marker.origin_id)
            if not r:
                continue
            if marker.type in (MarkerType.PLANNER, MarkerType.ACTIVATION):
                r.inject_energy(marker.energy // 20, self.tick_count, cap=self.calibration.energy_cap)
            node_markers.setdefault(marker.origin_id, []).append(marker)
            for tgt_id, packed in r.connections.items():
                w, et = unpack_edge(packed)
                if et == EDGE_EXCEPT:
                    continue
                decay = 0.85 if marker.type == MarkerType.ACTIVATION else 0.65
                new_energy = int(marker.energy * (w / 255.0) * decay)
                if new_energy < 5:
                    continue
                new_marker = Marker(
                    type=marker.type,
                    origin_id=tgt_id,
                    energy=new_energy,
                    payload=marker.payload.copy(),
                    step=marker.step + 1,
                    color=marker.color
                )
                next_queue.append(new_marker)
        for nid, markers in node_markers.items():
            types_present = set(m.type for m in markers)
            if MarkerType.SEEKER in types_present and MarkerType.PLANNER in types_present:
                collisions.append(nid)
                if nid in self.resonators:
                    self.resonators[nid].inject_energy(50, self.tick_count)
                    self.interference_log.append(
                        f"Такт {self.tick_count}: 💡 Marker Collision (Инсайт) в '{self.resonators[nid].label}'"
                    )
        self.marker_queue = next_queue
        return collisions
    
    def evoke_frames(self):
        """
        Frame Semantics v5: Топологическая эвокация фреймов.
        """
        for frame in self.active_frames.values():
            frame.activation *= 0.70
            for slot in frame.slots.values():
                slot.activation *= 0.70
                if slot.activation < 0.10:
                    slot.filler_id = None
                    slot.activation = 0.0
            if frame.activation < 0.10:
                frame.activation = 0.0
        
        for r in self.resonators.values():
            if not r.is_active(): continue
            for tgt_id, packed in r.connections.items():
                w, et = unpack_edge(packed)
                if et == EDGE_CAUSE and w > 30 and tgt_id in self.resonators:
                    tgt_r = self.resonators[tgt_id]
                    if tgt_r.is_active():
                        for frame in self.active_frames.values():
                            if frame.label == "CAUSATION":
                                act = min(1.0, (r.energy + tgt_r.energy + w) / 2000.0)
                                if act > frame.activation:
                                    frame.activation = act
                                    frame.bind_slot("Cause", r.id, act)
                                    frame.bind_slot("Effect", tgt_id, act)
        
        ADJ_ENDINGS = ('ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие')
        for r in self.resonators.values():
            if not r.is_active(): continue
            props = []
            for tgt_id, packed in r.connections.items():
                w, et = unpack_edge(packed)
                if et == EDGE_SYNTAGM and w > 20 and tgt_id in self.resonators:
                    tgt_r = self.resonators[tgt_id]
                    if tgt_r.is_active():
                        is_adj = tgt_r.label.endswith(ADJ_ENDINGS)
                        cause_out = sum(1 for p in tgt_r.connections.values() if unpack_edge(p)[1] == EDGE_CAUSE)
                        if is_adj or cause_out == 0:
                            props.append((tgt_id, w))
            if props:
                props.sort(key=lambda x: x[1], reverse=True)
                best_prop_id = props[0][0]
                for frame in self.active_frames.values():
                    if frame.label == "STATE":
                        act = min(1.0, (r.energy + self.resonators[best_prop_id].energy) / 1500.0)
                        if act > frame.activation:
                            frame.activation = act
                            frame.bind_slot("Entity", r.id, act)
                            frame.bind_slot("Property", best_prop_id, act)
    
    def build_dependency_tree(self, seed_ids: List[int]) -> Optional[DependencyNode]:
        """
        Строит дерево зависимостей из концептуального плана и активных фреймов.
        """
        if not seed_ids:
            return None
        best_frame = None
        max_act = 0.20
        for frame in self.active_frames.values():
            if frame.activation > max_act and len(frame.get_filled_roles()) >= 2:
                max_act = frame.activation
                best_frame = frame
        if best_frame:
            filled = best_frame.get_filled_roles()
            if best_frame.label == "CAUSATION" and "Cause" in filled and "Effect" in filled:
                root_node = DependencyNode(node_id=filled["Cause"], role="ROOT")
                effect_node = DependencyNode(node_id=filled["Effect"], role="EFFECT")
                root_node.children.append(effect_node)
                return root_node
            elif best_frame.label == "STATE" and "Entity" in filled and "Property" in filled:
                root_node = DependencyNode(node_id=filled["Entity"], role="ROOT")
                prop_node = DependencyNode(node_id=filled["Property"], role="PROP")
                root_node.children.append(prop_node)
                return root_node
        active_seeds = [(sid, self.resonators[sid].energy) for sid in seed_ids if sid in self.resonators]
        if not active_seeds:
            return None
        active_seeds.sort(key=lambda x: x[1], reverse=True)
        root_id = active_seeds[0][0]
        root_node = DependencyNode(node_id=root_id, role="ROOT")
        visited = {root_id}
        r = self.resonators[root_id]
        neighbors = []
        for tgt_id, packed in r.connections.items():
            if tgt_id in self.resonators and tgt_id not in visited:
                w, et = unpack_edge(packed)
                if et != EDGE_EXCEPT:
                    neighbors.append((tgt_id, w, et))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        for tgt_id, w, et in neighbors[:3]:
            role = "OBJ"
            if et == EDGE_CAUSE: role = "EFFECT"
            elif et == EDGE_EFFECT: role = "CAUSE"
            elif et == EDGE_SYNTAGM:
                tgt_lbl = self.resonators[tgt_id].label
                if tgt_lbl.endswith(('ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие')):
                    role = "PROP"
            child = DependencyNode(node_id=tgt_id, role=role)
            root_node.children.append(child)
            visited.add(tgt_id)
        return root_node
    
    def build_conceptual_plan(self, seed_node_ids: List[int], horizon: int = 7) -> List[int]:
        """
        Message Planning & Microplanning.
        """
        if not seed_node_ids:
            return []
        path_ids = self.tick_backward(seed_node_ids[0], max_depth=horizon, prefer_syn=True)
        for sid in seed_node_ids:
            self.inject_marker(Marker(
                type=MarkerType.PLANNER,
                origin_id=sid,
                energy=100,
                payload={"intent": "EXPLAIN"},
                color=sid
            ))
        for _ in range(2):
            self.tick_markers(max_steps=2)
        plan = list(path_ids)
        visited = set(plan)
        active_nodes = [
            r for r in self.resonators.values()
            if r.energy > 50 and r.id not in visited and
               not r.label.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:'))
        ]
        active_nodes.sort(key=lambda x: x.energy, reverse=True)
        for r in active_nodes[:5]:
            if len(plan) >= horizon:
                break
            is_connected = False
            for pid in plan:
                if pid in self.resonators:
                    if pid in r.connections or r.id in self.resonators[pid].connections:
                        is_connected = True
                        break
            if is_connected:
                plan.append(r.id)
                visited.add(r.id)
        self.blackboard.conceptual_plan = plan
        return plan
    
    # 🆕 Приоритет 1.8: Оценка качества плана
    def score_plan_quality(self, plan_ids: List[int], seed_ids: List[int]) -> dict:
        """
        Оценка качества концептуального плана с Бритвой Оккама.
        """
        if not plan_ids:
            return {"score": 0.0, "coverage": 0.0, "connectivity": 0.0, "causality": 0.0, "complexity_penalty": 1.0, "warnings": ["Пустой план"]}
        
        # 1. Покрытие
        plan_set = set(plan_ids)
        seed_set = set(seed_ids)
        covered = len(plan_set & seed_set)
        coverage = covered / len(seed_set) if seed_set else 0.0
        
        # 2. Связность
        connected_pairs = 0
        total_pairs = 0
        for i in range(len(plan_ids) - 1):
            id1, id2 = plan_ids[i], plan_ids[i+1]
            if id1 in self.resonators and id2 in self.resonators:
                total_pairs += 1
                r1, r2 = self.resonators[id1], self.resonators[id2]
                if id2 in r1.connections or id1 in r2.connections:
                    connected_pairs += 1
        connectivity = connected_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # 3. Причинность
        causal_count = 0
        for i in range(len(plan_ids) - 1):
            id1, id2 = plan_ids[i], plan_ids[i+1]
            if id1 in self.resonators and id2 in self.resonators:
                r1 = self.resonators[id1]
                if id2 in r1.connections:
                    _, et = unpack_edge(r1.connections[id2])
                    if et in (EDGE_CAUSE, EDGE_EFFECT, EDGE_COND):
                        causal_count += 1
        causality = causal_count / len(plan_ids) if plan_ids else 0.0
        
        # 4. Гипотетичность (штраф)
        hypothesis_count = sum(1 for pid in plan_ids if pid in self.hypotheses)
        hypothesis_penalty = hypothesis_count / len(plan_ids) if plan_ids else 0.0
        
        # 5. Сложность (Бритва Оккама)
        complexity = len(plan_ids)
        complexity_penalty = 0.90 ** max(0, complexity - 3)
        
        # 6. Конфликтность (штраф)
        contradiction_count = sum(1 for pid in plan_ids if pid in self.resonators and self.resonators[pid].belief_status == "contradiction")
        contradiction_penalty = contradiction_count / len(plan_ids) if plan_ids else 0.0
        
        # Финальная формула
        score = (
            coverage * 2.0 +
            connectivity * 1.5 +
            causality * 1.0
        ) * complexity_penalty * (1.0 - hypothesis_penalty * 0.5) * (1.0 - contradiction_penalty * 0.8)
        
        warnings = []
        if hypothesis_penalty > 0.3:
            warnings.append(f"Много гипотез в плане ({hypothesis_count})")
        if contradiction_penalty > 0.2:
            warnings.append(f"Есть противоречия в плане ({contradiction_count})")
        if complexity > 5:
            warnings.append(f"План переусложнен (длина {complexity})")
        
        return {
            "score": score,
            "coverage": coverage,
            "connectivity": connectivity,
            "causality": causality,
            "complexity_penalty": complexity_penalty,
            "warnings": warnings
        }
    
    def score_hypothesis(self, plan_ids: List[int], seed_ids: List[int]) -> float:
        """
        🧠 Оценка качества концептуального плана (Winner-Take-All).
        """
        if not plan_ids:
            return 0.0
        plan_set = set(plan_ids)
        seed_set = set(seed_ids)
        covered = len(plan_set & seed_set)
        coverage = covered / len(seed_set) if seed_set else 0.0
        connected_pairs = 0
        total_pairs = 0
        for i in range(len(plan_ids) - 1):
            id1, id2 = plan_ids[i], plan_ids[i+1]
            if id1 in self.resonators and id2 in self.resonators:
                total_pairs += 1
                r1, r2 = self.resonators[id1], self.resonators[id2]
                if id2 in r1.connections or id1 in r2.connections:
                    connected_pairs += 1
                else:
                    n1 = set(r1.connections.keys())
                    n2 = set(r2.connections.keys())
                    union = len(n1 | n2)
                    intersection = len(n1 & n2)
                    if union > 0 and (intersection / union) > 0.10:
                        connected_pairs += 1
        connectivity = connected_pairs / total_pairs if total_pairs > 0 else 0.0
        total_energy = sum(self.resonators[pid].energy for pid in plan_ids if pid in self.resonators)
        energy_score = min(1.0, total_energy / 5000.0)
        self.evoke_frames()
        frame_fill = 0.0
        active_frames_count = 0
        for frame in self.active_frames.values():
            if frame.activation > 0.20:
                active_frames_count += 1
                filled_slots = len(frame.get_filled_roles())
                total_slots = len(frame.slots)
                if total_slots > 0:
                    frame_fill += filled_slots / total_slots
        if active_frames_count > 0:
            frame_fill /= active_frames_count
        unique_ids = len(set(plan_ids))
        repeat_penalty = 1.0 - (unique_ids / len(plan_ids)) if plan_ids else 0.0
        score = (
            coverage * 2.0 +
            connectivity * 1.5 +
            energy_score * 1.0 +
            frame_fill * 1.5
        ) * (1.0 - repeat_penalty * 0.5)
        return score
    
    # ============================================================
    # 🧩 Раздел 27.2: Волна с причинной физикой
    # ============================================================
    def find_best_resonator_by_label(self, query: str) -> Optional['Resonator']:
        """🔮 Универсальный поиск узла по строке."""
        query = query.strip().lower()
        if not query: return None
        if query in self.label_to_id:
            return self.resonators[self.label_to_id[query]]
        if f"root:{query}" in self.label_to_id:
            return self.resonators[self.label_to_id[f"root:{query}"]]
        best_r = None
        best_score = -1
        for label, rid in self.label_to_id.items():
            if label == "SELF" or label.startswith(("EPOCH:", "mod:")): continue
            r = self.resonators[rid]
            stem = label.split(":", 1)[1] if ":" in label else label
            stem_low = stem.lower()
            score = 0
            if stem_low == query: score += 1000
            elif query in stem_low or stem_low in query: score += 500
            if score > 0:
                score += len(r.connections) * 2
                score += r.energy // 100
                if score > best_score:
                    best_score = score
                    best_r = r
        return best_r
    
    def tick(self):
        self.tick_count += 1
        self.tick_markers(max_steps=2)
        self.evoke_frames()
        transfers: Dict[int, Tuple[int, List[int]]] = {}
        for r in self.resonators.values():
            conn_count = len(r.connections)
            if r.is_active():
                # 🆕 Приоритет 1.5: Гипотезы с пониженным весом
                energy_mult = 0.3 if r.id in self.hypotheses else 1.0
                for target_id, packed_edge in r.connections.items():
                    weight, edge_type = unpack_edge(packed_edge)
                    if edge_type == EDGE_EXCEPT:
                        continue
                    effective_weight = weight
                    if edge_type == EDGE_ANALOG:
                        effective_weight = weight // 2
                    if edge_type == EDGE_GOAL:
                        effective_weight = min(MASK_WEIGHT, weight + (weight // 2))
                    if edge_type == EDGE_IS_A:
                        effective_weight = min(MASK_WEIGHT, weight + (weight // 4))  # 🆕 Усиление классовых связей
                    transferred = int((r.energy * effective_weight * energy_mult) // 256)
                    if conn_count > 5 and transferred > 0:
                        transferred = (transferred * 5) // conn_count
                    # 🆕 Жёсткий потолок на одну передачу за такт (перебалансировка энергии)
                    transferred = min(transferred, 200)
                    if transferred > 0:
                        if target_id not in transfers:
                            transfers[target_id] = (0, [])
                        e, sources = transfers[target_id]
                        transfers[target_id] = (e + transferred, sources + [r.id])
            if getattr(self, 'self_id', -1) != r.id:
                r.decay(self.decay_percent)
        for rid, (e, sources) in transfers.items():
            if rid in self.resonators:
                target_r = self.resonators[rid]
                unique_sources = list(set(sources))
                target_r.inject_energy(e, self.tick_count, source_ids=unique_sources)
                if target_r.state == TruthValue.VOID and target_r.is_active():
                    target_r.state = TruthValue.TRUE
                    self.apply_defeater(target_r.id)
                if target_r.is_active():
                    for src_id in unique_sources:
                        if src_id in self.resonators:
                            src_r = self.resonators[src_id]
                            old_p = src_r.connections.get(rid, 0)
                            old_w, old_t = unpack_edge(old_p)
                            if old_t == EDGE_SYNTAGM:
                                src_r.connections[rid] = pack_edge(min(255, old_w + 3), EDGE_SYNTAGM)
                            old_p2 = target_r.connections.get(src_id, 0)
                            old_w2, old_t2 = unpack_edge(old_p2)
                            if old_t2 == EDGE_SYNTAGM:
                                target_r.connections[src_id] = pack_edge(min(255, old_w2 + 3), EDGE_SYNTAGM)
                if len(unique_sources) >= 2 and e > 30:
                    src_labels = [self.resonators[s].label for s in unique_sources if s in self.resonators]
                    insight = f"Такт {self.tick_count}: 💡 Озарение в '{target_r.label}' от {src_labels}"
                    self.interference_log.append(insight)
                    target_r.inject_energy(15, self.tick_count)
        for r in list(self.resonators.values()):
            if r.is_active() and r.state == TruthValue.TRUE and r.defeats:
                self.apply_defeater(r.id)
        if len(self.interference_log) > 10:
            self.interference_log = self.interference_log[-10:]
        if len(self.paradox_log) > 10:
            self.paradox_log = self.paradox_log[-10:]
    
    def extract_spore(self, top_k_attractors: int = 3) -> 'Spore':
        """
        🧬 v7.1: Извлекает ТОЛЬКО ядро идентичности.
        """
        spore = Spore()
        label_id_map = {}
        import math
        attractors = []
        for r in self.resonators.values():
            if r.label == 'SELF': continue
            if r.label.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')): continue
            if '->' in r.label or '+' in r.label: continue
            if not r.label.startswith('root:'): continue
            gravity = r.energy * math.log2(1 + len(r.connections))
            attractors.append((gravity, r))
        attractors.sort(key=lambda x: x[0], reverse=True)
        top_attractors = attractors[:top_k_attractors]
        spore_node_ids = set()
        for _, r in top_attractors:
            spore_node_ids.add(r.id)
        for node_id in spore_node_ids:
            if node_id in self.resonators:
                spore.add_resonator(self.resonators[node_id], label_id_map)
        for node_id in spore_node_ids:
            if node_id in self.resonators:
                r = self.resonators[node_id]
                src_label_id = label_id_map.get(r.label)
                if src_label_id is not None:
                    for tgt_id, packed_edge in r.connections.items():
                        if tgt_id in spore_node_ids and tgt_id in self.resonators:
                            tgt_r = self.resonators[tgt_id]
                            tgt_label_id = label_id_map.get(tgt_r.label)
                            if tgt_label_id is not None:
                                spore.connections.append((src_label_id, tgt_label_id, packed_edge))
        spore.parent_tick = self.tick_count
        return spore
    
    def enter_dormancy(self):
        """Переводит кристалл в режим Проводника с высоким трением."""
        self.is_dormant = True
        self.genome &= ~(0xFFFFFFFF << GENE_DECAY_SHIFT)
        self.genome |= (80 << GENE_DECAY_SHIFT)
        for r in self.resonators.values():
            r.context_sources.clear()
            r.context_mask = 0
        self.skills.clear()
    
    def do_intervention(self, label: str, energy: int = 500) -> List[str]:
        r = self.get_or_create(label)
        log = [f"🔬 do('{label}'): Вмешательство начато"]
        severed = 0
        for src_r in self.resonators.values():
            if r.id in src_r.connections:
                packed = src_r.connections[r.id]
                w, et = unpack_edge(packed)
                if et == EDGE_CAUSE:
                    src_r.connections[r.id] = pack_edge(0, EDGE_SYNTAGM)
                    log.append(f"  ✂️ Отрезана CAUSE-связь от '{src_r.label}'")
                    severed += 1
        r.state = TruthValue.VOID
        r.energy = 0
        r.inject_energy(self.calibration.concept_inject_energy, self.tick_count, cap=self.calibration.energy_cap)
        r.state = TruthValue.TRUE
        log.append(f"  ⚡ Инжектировано {energy} энергии (независимо от причин)")
        log.append(f"  📊 Отрезано {severed} причинных связей")
        return log
    
    def find_analogies(self, label: str) -> List[Tuple[str, int]]:
        if label not in self.label_to_id:
            return []
        r = self.resonators[self.label_to_id[label]]
        analogies = []
        for target_id, packed in r.connections.items():
            w, et = unpack_edge(packed)
            if et == EDGE_ANALOG and target_id in self.resonators:
                analogies.append((self.resonators[target_id].label, w))
        analogies.sort(key=lambda x: x[1], reverse=True)
        return analogies
    
    def infer_causal_links(self, min_co_activation: int = 3) -> List[str]:
        log = []
        active_nodes = [r for r in self.resonators.values() if r.is_active()]
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                a, b = active_nodes[i], active_nodes[j]
                if b.id in a.connections:
                    w, et = unpack_edge(a.connections[b.id])
                    if et == EDGE_SYNTAGM and w > 50:
                        a.connections[b.id] = pack_edge(w, EDGE_CAUSE)
                        b.connections[a.id] = pack_edge(w, EDGE_EFFECT)
                        log.append(f"🔗 Причинный вывод: '{a.label}' →CAUSE '{b.label}' (вес={w})")
        return log
    
    # --- Обратная волна (Вектор 5) ---
    def tick_backward(self, vacuum_id: int, max_depth: int = None, prefer_syn: bool = False) -> List[int]:
        """
        Обратная трассировка с Семантической Гравитацией v2 + Marker Passing.
        """
        if max_depth is None:
            max_depth = self.calibration.backward_max_depth
        if vacuum_id not in self.resonators:
            return [vacuum_id]
        vac_r = self.resonators[vacuum_id]
        vac_hdc = vac_r.hdc_vector
        vac_ctx = getattr(vac_r, 'context_mask', 0)
        start_id = vacuum_id
        if not vac_r.connections:
            best_sim = -1.0
            best_anchor_id = None
            search_limit = min(len(self.resonators), self.calibration.lcs_search_limit * 2)
            candidates = [
                (rid, r) for rid, r in self.resonators.items()
                if r.connections and r.state != TruthValue.FALSE
            ][:search_limit]
            for rid, r in candidates:
                sim = self.encoder.similarity(vac_hdc, r.hdc_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_anchor_id = rid
            if best_anchor_id is not None and best_sim >= self.calibration.query_min_similarity:
                start_id = best_anchor_id
            else:
                return [vacuum_id]
        path: List[int] = [start_id]
        visited: set = {start_id}
        current_id = start_id
        cal = self.calibration
        w_cause   = cal.backward_cause_mult
        w_effect  = cal.backward_effect_mult
        w_cond    = cal.backward_cond_mult
        w_syn     = cal.backward_syn_mult
        bonus_true = cal.backward_true_state_bonus
        bonus_abs  = cal.backward_abstract_bonus
        anchor_min = cal.backward_vacuum_anchor_sim
        _has_except = hasattr(self, '_EDGE_EXCEPT') or 'EDGE_EXCEPT' in globals()
        for depth in range(max_depth):
            current_r = self.resonators.get(current_id)
            if not current_r or not current_r.connections:
                break
            best_score = -float('inf')
            best_neighbor = None
            cur_ctx = getattr(current_r, 'context_mask', 0)
            for neighbor_id, packed_edge in current_r.connections.items():
                if neighbor_id in visited:
                    continue
                neighbor = self.resonators.get(neighbor_id)
                if neighbor is None:
                    continue
                if neighbor.state == TruthValue.FALSE:
                    continue
                weight, edge_type = unpack_edge(packed_edge)
                is_abstract = neighbor.label.startswith(('mdl:', 'cluster:'))
                if _has_except and edge_type == EDGE_EXCEPT:
                    continue
                if edge_type == EDGE_CAUSE:
                    type_mult = w_cause
                elif edge_type == EDGE_EFFECT:
                    type_mult = w_effect
                elif edge_type == EDGE_COND:
                    type_mult = w_cond
                elif edge_type == EDGE_SYNTAGM:
                    type_mult = w_syn
                else:
                    type_mult = 1.0
                effective_weight = weight
                if prefer_syn and edge_type == EDGE_SYNTAGM:
                    effective_weight = int(weight * 1.5)
                if edge_type == EDGE_SYNTAGM:
                    sim_to_vac = self.encoder.similarity(vac_hdc, neighbor.hdc_vector)
                    decayed_threshold = anchor_min + (depth * 0.03)
                    if sim_to_vac < decayed_threshold:
                        continue
                neighbor_ctx = getattr(neighbor, 'context_mask', 0)
                context_penalty = 0
                if cur_ctx and neighbor_ctx and (cur_ctx != neighbor_ctx):
                    context_penalty = -200
                semantic_sim = self.encoder.similarity(vac_hdc, neighbor.hdc_vector)
                if edge_type == EDGE_SYNTAGM:
                    score = int(semantic_sim * 180) + (effective_weight * 2)
                else:
                    score = int(semantic_sim * 120) + int(effective_weight * type_mult * 0.25)
                score += neighbor.energy // 100
                if neighbor.state == TruthValue.TRUE:
                    score += bonus_true
                if is_abstract:
                    score += bonus_abs
                depth_penalty = depth * 4
                score -= depth_penalty
                score += context_penalty
                if semantic_sim < anchor_min and depth > 0:
                    score -= 70
                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor_id
            if best_neighbor is None:
                break
            if best_score < 0 and depth > 2:
                break
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current_id = best_neighbor
        if len(path) > 1:
            self.inject_marker(Marker(
                type=MarkerType.SEEKER,
                origin_id=vacuum_id,
                energy=100,
                payload={"intent": "FILL_GAP"},
                color=vacuum_id
            ))
        return path
    
    def inject_path_energy(self, path: List[int], energy: int = 300):
        for node_id in path:
            if node_id in self.resonators:
                self.resonators[node_id].inject_energy(energy, self.tick_count, cap=self.calibration.energy_cap)
    
    def etch_skill(self, wave_path: List[int], insight_node: int):
        vectors = [self.resonators[nid].hdc_vector for nid in wave_path if nid in self.resonators]
        if insight_node in self.resonators:
            vectors.append(self.resonators[insight_node].hdc_vector)
        if not vectors:
            return
        skill_vector = self.encoder.bundle(vectors)
        skill_label = f"skill:{len(self.skills)}"
        new_id = self._next_id
        self._next_id += 1
        skill_r = Resonator(id=new_id, label=skill_label, hdc_vector=skill_vector)
        skill_r.state = TruthValue.TRUE
        self.resonators[new_id] = skill_r
        self.label_to_id[skill_label] = new_id
        self.skills.append(new_id)
    
    def _get_neighbor_view(self, resonator: Resonator, min_weight: int = 10) -> Dict[int, Tuple[Resonator, int, int]]:
        """Возвращает 1-hop окружение узла."""
        view: Dict[int, Tuple[Resonator, int, int]] = {}
        for tgt_id, packed in resonator.connections.items():
            if tgt_id not in self.resonators:
                continue
            weight, edge_type = unpack_edge(packed)
            if weight > min_weight:
                view[tgt_id] = (self.resonators[tgt_id], weight, edge_type)
        return view
    
    def _get_hubs(self, exclude_ids: Optional[Set[int]] = None) -> Set[int]:
        """Определяет узлы-хабы."""
        exclude_ids = set(exclude_ids or set())
        degrees: List[Tuple[int, int]] = []
        for resonator in self.resonators.values():
            if resonator.id in exclude_ids:
                continue
            label = resonator.label or ""
            if label == "SELF" or label.startswith(("EPOCH:", "mod:", "mdl:", "cluster:", "skill:")):
                continue
            degrees.append((len(resonator.connections), resonator.id))
        if not degrees:
            return set()
        degrees.sort(reverse=True)
        degree_values = [degree for degree, _ in degrees]
        degree_threshold = max(median(degree_values), degrees[min(len(degrees) - 1, int(len(degrees) * 0.15))][0])
        return {resonator_id for degree, resonator_id in degrees if degree >= degree_threshold}
    
    def _jaccard_similarity(self, left_ids: Set[int], right_ids: Set[int]) -> float:
        if not left_ids and not right_ids:
            return 0.0
        union = left_ids | right_ids
        return len(left_ids & right_ids) / len(union) if union else 0.0
    
    def _expand_context_ids(self, seed_ids: Set[int], depth: int = 2) -> Set[int]:
        """Расширяет набор узлов по графу."""
        if not seed_ids:
            return set()
        expanded = set(seed_ids)
        frontier = list(seed_ids)
        for _ in range(depth):
            next_frontier: Set[int] = set()
            for node_id in frontier:
                if node_id not in self.resonators:
                    continue
                resonator = self.resonators[node_id]
                for tgt_id in resonator.connections.keys():
                    if tgt_id in self.resonators:
                        next_frontier.add(tgt_id)
            expanded.update(next_frontier)
            frontier = list(next_frontier)
        return expanded
    
    def _find_antagonisms(self, unique1_ids: Set[int], unique2_ids: Set[int], r1: Resonator, r2: Resonator,
                          hubs: Set[int], n1: Dict[int, Tuple[Resonator, int, int]],
                          n2: Dict[int, Tuple[Resonator, int, int]], debug: bool = False) -> List[Tuple[str, str]]:
        """Ищет антагонизмы."""
        antagonisms: List[Tuple[str, str]] = []
        seen_pairs: Set[Tuple[int, int]] = set()
        def emit_debug(message: str):
            if debug:
                print(f"   🐛 [COMPARE] {message}")
        def display_label(label: str) -> str:
            label = (label or '').strip()
            if not label:
                return label
            for prefix in ("root:", "cluster:", "mdl:", "skill:", "mod:", "EPOCH:"):
                if label.startswith(prefix):
                    return label[len(prefix):]
            return label
        for u1_id in unique1_ids:
            for u2_id in unique2_ids:
                pair = tuple(sorted((u1_id, u2_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                u1_r = self.resonators.get(u1_id)
                u2_r = self.resonators.get(u2_id)
                if not u1_r or not u2_r or u1_r.id == u2_r.id:
                    continue
                l1 = u1_r.label
                l2 = u2_r.label
                if l1 == l2:
                    continue
                direct_except = False
                if u2_id in u1_r.connections:
                    _, edge_type = unpack_edge(u1_r.connections[u2_id])
                    direct_except = edge_type == EDGE_EXCEPT
                if not direct_except and u1_id in u2_r.connections:
                    _, edge_type = unpack_edge(u2_r.connections[u1_id])
                    direct_except = edge_type == EDGE_EXCEPT
                structural_strength = len(u1_r.connections) + len(u2_r.connections)
                root_like = u1_r.label.startswith("root:") or u2_r.label.startswith("root:")
                weight1 = n1.get(u1_id, (None, 0, 0))[1] if u1_id in n1 else 0
                weight2 = n2.get(u2_id, (None, 0, 0))[1] if u2_id in n2 else 0
                direct_strength = weight1 + weight2
                shared_context = (set(u1_r.connections.keys()) & set(u2_r.connections.keys())) - hubs - {r1.id, r2.id}
                direct_trait = False
                emit_debug(
                    f"candidate '{display_label(l1)}' vs '{display_label(l2)}' "
                    f"direct_except={direct_except} direct_strength={direct_strength} "
                    f"structural_strength={structural_strength} root_like={root_like} "
                    f"hdc_sim={self.encoder.similarity(u1_r.hdc_vector, u2_r.hdc_vector):.3f}"
                )
                # Требование: ТОЛЬКО прямая EXCEPT-связь или очень сильное структурное доказательство
                if not direct_except:
                    # Без прямой EXCEPT-связи антагонизм не создаём
                    # (убираем эвристики "shared context" и "hdc_sim < 0.56")
                    continue
                if self._is_false_antagonism(l1, l2):
                    emit_debug("skip: user-marked false antagonism")
                    continue
                emit_debug("accept: direct EXCEPT edge")
                antagonisms.append((l1, l2))
                if len(antagonisms) >= 15:
                    break
            if len(antagonisms) >= 15:
                break
        return antagonisms[:15]
    
    # ============================================================
    # 🧩 Раздел 27.3: Топологическое Сравнение (Фазы B1, B2, B3)
    # ============================================================
    def compare_entities(self, label1: str, label2: str, debug: bool = False) -> dict:
        """
        Сравнивает два концепта по их топологическому окружению.
        """
        def display_label(label: str) -> str:
            label = (label or '').strip()
            if not label:
                return label
            for prefix in ("root:", "cluster:", "mdl:", "skill:", "mod:", "EPOCH:"):
                if label.startswith(prefix):
                    return label[len(prefix):]
            return label
        def resolve_best_node(l):
            l = (l or '').strip()
            if not l:
                return None
            normalized = l.lower()
            best_resonator = None
            best_score = -1
            for label_key, node_id in self.label_to_id.items():
                resonator = self.resonators[node_id]
                if resonator.label == "SELF":
                    continue
                label_text = resonator.label.lower()
                stem = resonator.label
                if resonator.label.startswith(("root:", "cluster:", "mdl:", "skill:", "mod:", "EPOCH:")):
                    stem = resonator.label.split(":", 1)[1]
                stem_low = stem.lower()
                score = 0
                if label_text == normalized:
                    score += 300
                if label_text == f"root:{normalized}":
                    score += 1200
                if stem_low == normalized:
                    score += 900
                if stem_low and (stem_low in normalized or normalized in stem_low):
                    score += 400
                if resonator.label.startswith("root:"):
                    score += 150
                score += len(resonator.connections) * 3
                score += resonator.energy // 120
                if score > best_score:
                    best_score = score
                    best_resonator = resonator
            return best_resonator
        r1 = resolve_best_node(label1)
        r2 = resolve_best_node(label2)
        if not r1 or not r2:
            return {"error": "Один или оба концепта не найдены в памяти."}
        if r1.id == r2.id:
            return {"error": "Это один и тот же концепт."}
        n1 = self._get_neighbor_view(r1)
        n2 = self._get_neighbor_view(r2)
        if debug:
            print(f"   🐛 [COMPARE] resolved '{label1}' -> '{r1.label}' (id={r1.id})")
            print(f"   🐛 [COMPARE] resolved '{label2}' -> '{r2.label}' (id={r2.id})")
            print(f"   🐛 [COMPARE] neighbor view sizes: {len(n1)} / {len(n2)}")
            for source_name, view in ((label1, n1), (label2, n2)):
                if view:
                    sample = []
                    for node_id, (res, weight, edge_type) in sorted(view.items(), key=lambda item: item[1][1], reverse=True)[:8]:
                        sample.append(f"{self.resonators[node_id].label} w={weight} t={edge_type_name(pack_edge(weight, edge_type))}")
                    print(f"   🐛 [COMPARE] {source_name} neighbors: {', '.join(sample)}")
        common_ids = set(n1.keys()) & set(n2.keys())
        hubs = self._get_hubs(exclude_ids={r1.id, r2.id})
        service_ids = {self.self_id}
        for label in self.label_to_id:
            if label.startswith(("EPOCH:", "mod:", "mdl:", "cluster:", "skill:")):
                service_ids.add(self.label_to_id[label])
        common_traits = []
        name1 = display_label(r1.label).lower()
        name2 = display_label(r2.label).lower()
        for cid in common_ids:
            if cid in service_ids or cid in hubs:
                continue
            resonator = self.resonators[cid]
            if resonator.label.startswith(("mdl:", "cluster:", "skill:", "mod:", "EPOCH:")):
                continue
            if '->' in resonator.label or '+' in resonator.label:
                continue
            lbl = display_label(resonator.label)
            if name1 in lbl.lower() or name2 in lbl.lower():
                continue
            if len(lbl) >= 3 and lbl != 'SELF':
                common_traits.append(lbl)
        if not common_traits:
            for cid in set(n1.keys()) | set(n2.keys()):
                if cid in service_ids or cid in hubs or cid in {r1.id, r2.id}:
                    continue
                resonator = self.resonators[cid]
                if resonator.label.startswith("root:"):
                    continue
                if cid in n1 and cid in n2:
                    continue
                if cid in n1:
                    other = n2
                elif cid in n2:
                    other = n1
                else:
                    continue
                if not other:
                    continue
                best_other_id = None
                best_sim = -1.0
                for other_id in other.keys():
                    if other_id in service_ids or other_id in hubs or other_id in {r1.id, r2.id}:
                        continue
                    other_res = self.resonators[other_id]
                    sim = self.encoder.similarity(resonator.hdc_vector, other_res.hdc_vector)
                    if sim > best_sim:
                        best_sim = sim
                        best_other_id = other_id
                if best_other_id is not None and best_sim > 0.65:
                    common_traits.append(display_label(resonator.label))
                    break
        synonym_traits = []
        for u1_id in set(n1.keys()) - {r1.id, r2.id}:
            u1_r = self.resonators[u1_id]
            for u2_id in set(n2.keys()) - {r1.id, r2.id}:
                u2_r = self.resonators[u2_id]
                if u1_r.id == u2_r.id:
                    continue
                if u1_id in hubs or u2_id in hubs:
                    continue
                hdc_sim = self.encoder.similarity(u1_r.hdc_vector, u2_r.hdc_vector)
                neighbor_jaccard = self._jaccard_similarity(set(u1_r.connections.keys()), set(u2_r.connections.keys()))
                if hdc_sim > 0.75 and neighbor_jaccard > 0.25:
                    synonym_traits.append(display_label(u1_r.label))
                    break
            if synonym_traits:
                break
        unique1_ids = set(n1.keys()) - set(n2.keys()) - {r1.id, r2.id} - service_ids
        unique2_ids = set(n2.keys()) - set(n1.keys()) - {r1.id, r2.id} - service_ids
        contrast_candidates = self._expand_context_ids(set(n1.keys()) | set(n2.keys()), depth=2)
        contrast_candidates -= service_ids
        contrast_candidates.discard(r1.id)
        contrast_candidates.discard(r2.id)
        antagonisms = self._find_antagonisms(unique1_ids, unique2_ids, r1, r2, hubs, n1, n2, debug=debug)
        if not antagonisms:
            for u1_id in contrast_candidates:
                if u1_id in service_ids or u1_id in {r1.id, r2.id}:
                    continue
                if u1_id not in self.resonators:
                    continue
                u1_r = self.resonators[u1_id]
                for u2_id in contrast_candidates:
                    if u2_id in service_ids or u2_id in {r1.id, r2.id} or u2_id == u1_id:
                        continue
                    if u2_id not in self.resonators:
                        continue
                    u2_r = self.resonators[u2_id]
                    if u1_r.id == u2_r.id:
                        continue
                    has_except = False
                    if u2_id in u1_r.connections:
                        _, edge_type = unpack_edge(u1_r.connections[u2_id])
                        has_except = edge_type == EDGE_EXCEPT
                    if not has_except and u1_id in u2_r.connections:
                        _, edge_type = unpack_edge(u2_r.connections[u1_id])
                        has_except = edge_type == EDGE_EXCEPT
                    if has_except:
                        antagonisms.append((display_label(u1_r.label), display_label(u2_r.label)))
                        break
                if antagonisms:
                    break
        common_traits = [trait for trait in common_traits if trait not in {display_label(r1.label), display_label(r2.label)}]
        common_traits = common_traits[:7]
        if synonym_traits:
            common_traits = [*synonym_traits, *common_traits]
        common_traits = list(dict.fromkeys(common_traits))[:7]
        clean_antagonisms = [
            (a1, a2) for a1, a2 in antagonisms
            if not self._is_false_antagonism(a1, a2)
        ]
        return {
            "entity1": display_label(label1) or r1.label,
            "entity2": display_label(label2) or r2.label,
            "common": common_traits,
            "antagonisms": clean_antagonisms[:5],
            "all_antagonisms": antagonisms,
            "unique1": [display_label(self.resonators[i].label) for i in list(unique1_ids)[:3]],
            "unique2": [display_label(self.resonators[i].label) for i in list(unique2_ids)[:3]]
        }
    
    def get_active(self, top_k: int = 10) -> List[Resonator]:
        active = [r for r in self.resonators.values() if r.energy > 0]
        active.sort(key=lambda x: x.energy, reverse=True)
        return active[:top_k]
    
    def get_questions(self, top_k: int = 3) -> List[Tuple['Resonator', float]]:
        threshold = self.entropy_threshold
        questions = []
        for r in self.resonators.values():
            if r.label == 'SELF' or r.label.startswith(('cluster:', 'mdl:', 'skill:', 'EPOCH:', 'mod:')):
                continue
            if r.energy > 15:
                causal_conns = 0
                for packed in r.connections.values():
                    _, et = unpack_edge(packed)
                    if et in (EDGE_CAUSE, EDGE_EFFECT, EDGE_COND):
                        causal_conns += 1
                entropy = (r.energy / 50.0) / (1.0 + causal_conns * 5.0)
                if entropy >= threshold:
                    questions.append((r, entropy))
        questions.sort(key=lambda x: x[1], reverse=True)
        return questions[:top_k]
    
    # --- Морфология (Вектор 3) ---
    def discover_cases(self) -> List[List[str]]:
        mods = [r for r in self.resonators.values() if r.label.startswith('mod:') and r.context_sources]
        if not mods:
            return []
        mod_contexts = {mod.id: set(mod.context_sources) for mod in mods}
        cases = []
        used = set()
        for i in range(len(mods)):
            if mods[i].id in used: continue
            cluster = [mods[i].label]
            set_i = mod_contexts[mods[i].id]
            for j in range(i + 1, len(mods)):
                if mods[j].id in used: continue
                set_j = mod_contexts[mods[j].id]
                union = len(set_i | set_j)
                if union == 0: continue
                jaccard = len(set_i & set_j) / union
                if jaccard > 0.45:
                    cluster.append(mods[j].label)
                    used.add(mods[j].id)
            if len(cluster) > 1:
                cases.append(cluster)
                used.add(mods[i].id)
        return cases
    
    # --- Геном (мутация) ---
    def mutate_genome(self, seed: int):
        state = seed & 0xFFFFFFFF
        mutation_mask = 0
        shifts = [GENE_DECAY_SHIFT, GENE_ENTROPY_THRESH, GENE_MAX_DEPTH]
        for shift in shifts:
            state = (state * 1103515245 + 12345) & 0xFFFFFFFF
            if (state % 100) < 15:
                bit_pos = state % 8
                mutation_mask |= (1 << (shift + bit_pos))
        self.genome ^= mutation_mask
        if self.decay_percent > 80:
            self.genome &= ~(0xFFFFFFFF << GENE_DECAY_SHIFT)
            self.genome |= (20 << GENE_DECAY_SHIFT)
    
    def discover_abstractions(self) -> List[str]:
        """Раздел 36: Сингамия — обнаружение топологического изоморфизма."""
        dreams = []
        attractors = [r for r in self.resonators.values()
                    if r.label.startswith('root:') and r.energy > self.calibration.syngrammy_min_attractor_energy]
        profiles = {}
        for attr in attractors:
            profile = []
            for tgt_id, packed in attr.connections.items():
                w, et = unpack_edge(packed)
                if w > self.calibration.syngrammy_min_edge_weight and tgt_id in self.resonators:
                    tgt = self.resonators[tgt_id]
                    category = "attr" if tgt.label.startswith('root:') else "prop"
                    tgt_degree = len(tgt.connections)
                    weight_bucket = min(w // 30, 3)
                    profile.append((et, category, weight_bucket, min(tgt_degree, 10)))
            profile.sort()
            profiles[attr.id] = (attr, profile)
        checked = set()
        for id1, (r1, p1) in profiles.items():
            for id2, (r2, p2) in profiles.items():
                if id1 >= id2: continue
                pair = tuple(sorted((id1, id2)))
                if pair in checked: continue
                checked.add(pair)
                if len(p1) >= self.calibration.syngrammy_min_profile_len and len(p2) >= self.calibration.syngrammy_min_profile_len:
                    set1 = set(p1)
                    set2 = set(p2)
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard = intersection / union if union > 0 else 0
                    if jaccard > self.calibration.syngrammy_jaccard_thresh:
                        abstract_label = f"mdl:{r1.label}+{r2.label}"
                        if abstract_label not in self.label_to_id:
                            # 🆕 Приоритет 1.10: Оценка поддержки
                            support = len([r for r in self.resonators.values()
                                           if r.id in r1.connections or r.id in r2.connections])
                            if support < 2:
                                dreams.append(f"⚠️ Отклонено: '{abstract_label}' (недостаточно поддержки: {support})")
                                continue
                            
                            bundled_hdc = self.encoder.bundle([r1.hdc_vector, r2.hdc_vector])
                            new_id = self._next_id
                            self._next_id += 1
                            abstract_r = Resonator(id=new_id, label=abstract_label, hdc_vector=bundled_hdc)
                            abstract_r.energy = (r1.energy + r2.energy) // 2
                            self.resonators[new_id] = abstract_r
                            self.label_to_id[abstract_label] = new_id
                            
                            # Создаём причину
                            reason = CrystalReason(
                                kind="sleep",
                                source_label=abstract_label,
                                source_type="sleep_abstraction",
                                confidence=0.4,
                                context=self.active_context,
                                timestamp=self.tick_count,
                                metadata={"support": support}
                            )
                            
                            self.connect(abstract_label, r1.label, weight=80, edge_type=EDGE_COND, reason=reason)
                            self.connect(abstract_label, r2.label, weight=80, edge_type=EDGE_COND, reason=reason)
                            
                            # Добавляем как гипотезу
                            self.add_hypothesis(new_id, reason)
                            
                            # Если поддержка высокая - автоматически подтверждаем
                            if support >= 3:
                                self.confirm_hypothesis(new_id)
                                dreams.append(f"🍎 Сингамия: '{abstract_label}' (Изоморфизм: {jaccard:.2f}, поддержка: {support}) ✓")
                            else:
                                dreams.append(f"🍎 Сингамия: '{abstract_label}' (Изоморфизм: {jaccard:.2f}, поддержка: {support}) [гипотеза]")
        return dreams
    
    # --- Сон ---
    def defragment(self):
        merged_count = 0
        pruned_count = 0
        dreams = []
        raw_nodes = {r.label: r for r in self.resonators.values() if not r.label.startswith(('root:', 'mod:', 'cluster:', 'mdl:', 'skill:'))}
        root_nodes = {r.label.replace('root:', ''): r for r in self.resonators.values() if r.label.startswith('root:')}
        to_delete_ids = []
        for raw_label, raw_r in raw_nodes.items():
            matched_root_label = None
            if raw_label in root_nodes:
                matched_root_label = raw_label
            else:
                for root_key in root_nodes:
                    if root_key in raw_label or raw_label in root_key:
                        if len(root_key) >= self.calibration.min_root_len:
                            matched_root_label = root_key
                            break
            if matched_root_label:
                root_r = root_nodes[matched_root_label]
                if raw_r.id != root_r.id:
                    for tgt_id, packed in list(raw_r.connections.items()):
                        if tgt_id != root_r.id and tgt_id in self.resonators:
                            w, et = unpack_edge(packed)
                            old_packed = root_r.connections.get(tgt_id, 0)
                            old_w, old_et = unpack_edge(old_packed)
                            new_w = min(MASK_WEIGHT, old_w + w) if (et == EDGE_SYNTAGM and old_et == EDGE_SYNTAGM) else max(old_w, w)
                            root_r.connections[tgt_id] = pack_edge(new_w, et)
                            tgt_r = self.resonators[tgt_id]
                            if raw_r.id in tgt_r.connections:
                                tgt_r.connections[root_r.id] = tgt_r.connections.pop(raw_r.id)
                    root_r.energy = min(self.calibration.energy_cap, root_r.energy + raw_r.energy)
                    to_delete_ids.append(raw_r.id)
        for rid in to_delete_ids:
            if rid in self.resonators:
                label = self.resonators[rid].label
                del self.resonators[rid]
                self.label_to_id.pop(label, None)
                pruned_count += 1
        dead_ids = [r.id for r in self.resonators.values() if r.energy == 0 and len(r.connections) == 0 and len(r.defeats) == 0 and not r.context_sources and r.label != 'SELF']
        for rid in dead_ids:
            label = self.resonators[rid].label
            del self.resonators[rid]
            self.label_to_id.pop(label, None)
            pruned_count += 1
            if len(dreams) < 5:
                dreams.append(f"🍂 Забыто: '{label}'")
        strong_bonds = []
        checked_pairs = set()
        for r in list(self.resonators.values()):
            for tgt_id, packed in list(r.connections.items()):
                w, _ = unpack_edge(packed)
                if w > 80 and tgt_id in self.resonators:
                    pair = tuple(sorted((r.id, tgt_id)))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        strong_bonds.append((w, r.id, tgt_id))
        strong_bonds.sort(key=lambda x: x[0], reverse=True)
        for weight, r_id, tgt_id in strong_bonds[:3]:
            r = self.resonators[r_id]
            target_r = self.resonators[tgt_id]
            cluster_label = f"cluster:{r.label}+{target_r.label}"
            if cluster_label not in self.label_to_id:
                # 🆕 Приоритет 1.10: Оценка поддержки
                support = len([node for node in self.resonators.values()
                              if r.id in node.connections or target_r.id in node.connections])
                if support < 2:
                    dreams.append(f"⚠️ Кластер отклонен: '{cluster_label}' (поддержка: {support})")
                    continue
                
                bundled_hdc = self.encoder.bundle([r.hdc_vector, target_r.hdc_vector])
                new_id = self._next_id
                self._next_id += 1
                cluster_r = Resonator(id=new_id, label=cluster_label, hdc_vector=bundled_hdc)
                cluster_r.energy = (r.energy + target_r.energy) // 2
                self.resonators[new_id] = cluster_r
                self.label_to_id[cluster_label] = new_id
                
                reason = CrystalReason(
                    kind="sleep",
                    source_label=cluster_label,
                    source_type="sleep_abstraction",
                    confidence=0.4,
                    context=self.active_context,
                    timestamp=self.tick_count,
                    metadata={"support": support}
                )
                
                self.connect(cluster_label, r.label, weight=50, reason=reason)
                self.connect(cluster_label, target_r.label, weight=50, reason=reason)
                
                self.add_hypothesis(new_id, reason)
                if support >= 3:
                    self.confirm_hypothesis(new_id)
                    dreams.append(f"🌌 Абстракция: '{cluster_label}' (поддержка: {support}) ✓")
                else:
                    dreams.append(f"🌌 Абстракция: '{cluster_label}' (поддержка: {support}) [гипотеза]")
        triads = {}
        for r in list(self.resonators.values()):
            for tgt_id, packed in r.connections.items():
                w1, _ = unpack_edge(packed)
                if w1 < self.calibration.mdl_triad_min_weight or tgt_id not in self.resonators: continue
                target = self.resonators[tgt_id]
                for tgt2_id, packed2 in target.connections.items():
                    w2, _ = unpack_edge(packed2)
                    if w2 < self.calibration.mdl_triad_min_weight or tgt2_id == r.id or tgt2_id not in self.resonators: continue
                    key = (r.id, tgt2_id)
                    triads[key] = triads.get(key, 0) + 1
        for (a_id, c_id), count in triads.items():
            if count >= self.calibration.mdl_triad_count_thresh:
                a_r = self.resonators[a_id]
                c_r = self.resonators[c_id]
                if a_r.energy < self.calibration.mdl_triad_min_energy or c_r.energy < self.calibration.mdl_triad_min_energy:
                    continue
                a_label = self.resonators[a_id].label
                c_label = self.resonators[c_id].label
                shortcut_label = f"mdl:{a_label}->{c_label}"
                if shortcut_label not in self.label_to_id:
                    # 🆕 Приоритет 1.10: Оценка поддержки
                    if count < 3:
                        dreams.append(f"⚠️ MDL отклонен: '{shortcut_label}' (поддержка: {count})")
                        continue
                    
                    a_r = self.resonators[a_id]
                    c_r = self.resonators[c_id]
                    bundled_hdc = self.encoder.bundle([a_r.hdc_vector, c_r.hdc_vector])
                    new_id = self._next_id
                    self._next_id += 1
                    shortcut_r = Resonator(id=new_id, label=shortcut_label, hdc_vector=bundled_hdc)
                    self.resonators[new_id] = shortcut_r
                    self.label_to_id[shortcut_label] = new_id
                    
                    reason = CrystalReason(
                        kind="sleep",
                        source_label=shortcut_label,
                        source_type="sleep_abstraction",
                        confidence=0.4,
                        context=self.active_context,
                        timestamp=self.tick_count,
                        metadata={"triad_count": count}
                    )
                    
                    self.connect(shortcut_label, a_label, weight=80, reason=reason)
                    self.connect(shortcut_label, c_label, weight=80, reason=reason)
                    
                    self.add_hypothesis(new_id, reason)
                    if count >= 5:
                        self.confirm_hypothesis(new_id)
                        dreams.append(f"🗜️ MDL: '{shortcut_label}' (×{count}) ✓")
                    else:
                        dreams.append(f"🗜️ MDL: '{shortcut_label}' (×{count}) [гипотеза]")
        potential_antagonists = []
        for r in list(self.resonators.values()):
            if r.label.startswith(('mdl:', 'cluster:', 'SELF', 'EPOCH:', 'mod:', 'skill:')): continue
            if r.energy < 50: continue
            base_ids = set()
            for tgt_id, packed in r.connections.items():
                w, et = unpack_edge(packed)
                if et in (EDGE_CAUSE, EDGE_COND, EDGE_SYNTAGM) and w >= 20:
                    base_ids.add(tgt_id)
            if len(base_ids) >= 2:
                potential_antagonists.append((r.id, base_ids))
        except_generated = 0
        for i in range(len(potential_antagonists)):
            id1, base1 = potential_antagonists[i]
            for j in range(i + 1, len(potential_antagonists)):
                id2, base2 = potential_antagonists[j]
                common_base = base1 & base2
                if len(common_base) >= 3:  # было >= 1
                    r1, r2 = self.resonators[id1], self.resonators[id2]
                    sim = self.encoder.similarity(r1.hdc_vector, r2.hdc_vector)
                    if sim < 0.30:  # только реально далёкие векторы
                        if id2 not in r1.connections or unpack_edge(r1.connections[id2])[1] != EDGE_EXCEPT:
                            self.connect(r1.label, r2.label, weight=self.calibration.except_antonym_weight, edge_type=EDGE_EXCEPT)
                            dreams.append(f"🛡 Авто-EXCEPT: '{r1.label}' ↔ '{r2.label}' (общая база: {len(common_base)})")
                            except_generated += 1
                            if except_generated > 10: break
            if except_generated > 10: break
        cases = self.discover_cases()
        for case_cluster in cases:
            dreams.append(f"📜 Падеж: {case_cluster}")
        annealed = self.anneal_paradoxes()
        if annealed > 0:
            dreams.append(f"🔥 Отжиг: {annealed} узлов мутировали.")
        antonyms_diverged = self._scan_for_antonyms_in_sleep(similarity_threshold=0.92, min_common_neighbors=5)
        if antonyms_diverged > 0:
            dreams.append(f"🎭 Структурализм: Разведено {antonyms_diverged} пар антонимов (Соссюр).")
        seed = self.tick_count ^ len(self.resonators)
        self.mutate_genome(seed)
        dreams.append(f"🧬 Геном: Decay={self.decay_percent}%, Thresh={self.entropy_threshold:.2f}")
        syn_dreams = self.discover_abstractions()
        dreams.extend(syn_dreams)
        return merged_count, pruned_count, dreams
    
    # 🆕 Приоритет 1.9: XAI-трассировка
    def explain_path(self, path_ids: List[int]) -> str:
        """XAI-отчёт для пути."""
        if not path_ids:
            return "Путь пуст."
        
        report = "=== XAI ТРАССИРОВКА ПУТИ ===\n"
        report += f"Длина пути: {len(path_ids)} узлов\n"
        report += f"Контекст: {self.active_context}\n\n"
        
        for i, node_id in enumerate(path_ids):
            if node_id not in self.resonators:
                report += f"❓ Узел {node_id} не найден\n"
                continue
            
            r = self.resonators[node_id]
            report += f"📍 Узел {i+1}: '{r.label}'\n"
            report += f"   Статус: {r.belief_status} (уверенность: {r.belief_confidence:.2f})\n"
            report += f"   Контекст: {r.belief_context}\n"
            
            if r.belief_reason:
                report += f"   Причина: {r.belief_reason.kind} ({r.belief_reason.source_type})\n"
                report += f"   Уверенность причины: {r.belief_reason.confidence:.2f}\n"
            
            # Связь с предыдущим узлом
            if i > 0:
                prev_id = path_ids[i-1]
                if prev_id in self.resonators:
                    prev_r = self.resonators[prev_id]
                    if node_id in prev_r.connections:
                        w, et = unpack_edge(prev_r.connections[node_id])
                        report += f"   ← Связь от '{prev_r.label}': {EDGE_NAMES.get(et, '?')} (вес={w})\n"
                        if edge_reason := self.edge_meta.get((prev_id, node_id)):
                            report += f"   Причина связи: {edge_reason.kind}\n"
            
            report += "\n"
        
        # Оценка качества
        quality = self.score_plan_quality(path_ids, [path_ids[0]] if path_ids else [])
        report += f"=== ОЦЕНКА КАЧЕСТВА ===\n"
        report += f"Покрытие: {quality['coverage']:.2f}\n"
        report += f"Связность: {quality['connectivity']:.2f}\n"
        report += f"Причинность: {quality['causality']:.2f}\n"
        report += f"Штраф Оккама: {quality['complexity_penalty']:.2f}\n"
        if quality['warnings']:
            report += f"⚠️ Предупреждения: {', '.join(quality['warnings'])}\n"
        report += f"========================\n"
        
        return report

# ============================================================
# 🧠 Раздел 28: ПЛАНИРОВЩИК И ТЕНЕВЫЕ ВОЛНЫ (v3.2)
# ============================================================
class ShadowLattice:
    """Zero-Copy клон решетки для симуляции будущего."""
    def __init__(self, base: 'CrystalLattice'):
        self.base = base
        self.overlay_energy: Dict[int, int] = {}
        self.tick = base.tick_count
    
    def get_energy(self, rid: int) -> int:
        if rid in self.overlay_energy:
            return self.overlay_energy[rid]
        r = self.base.resonators.get(rid)
        return r.energy if r else 0
    
    def set_energy(self, rid: int, val: int):
        # 🆕 Приоритет 0.5: Используем calibration вместо хардкода
        self.overlay_energy[rid] = max(0, min(self.base.calibration.shadow_energy_cap, val))
    
    def shadow_tick(self):
        self.tick += 1
        transfers: Dict[int, int] = {}
        active_ids = set(self.overlay_energy.keys())
        for rid, r in self.base.resonators.items():
            if r.energy > 15:
                active_ids.add(rid)
        for rid in active_ids:
            e = self.get_energy(rid)
            if e < 15: continue
            r_base = self.base.resonators.get(rid)
            if not r_base: continue
            self.set_energy(rid, int(e * 0.50))
            conn_count = len(r_base.connections)
            if conn_count == 0: continue
            energy_per_edge = e // conn_count
            for tgt_id, packed_edge in r_base.connections.items():
                weight, edge_type = unpack_edge(packed_edge)
                if edge_type == EDGE_EXCEPT: continue
                if edge_type == EDGE_ANALOG: weight = weight // 2
                if edge_type == EDGE_GOAL: weight = min(MASK_WEIGHT, weight + (weight // 2))
                transferred = (energy_per_edge * weight) // 256
                if transferred > 0:
                    transfers[tgt_id] = transfers.get(tgt_id, 0) + transferred
        for tgt_id, e in transfers.items():
            new_val = min(self.base.calibration.shadow_energy_cap, self.get_energy(tgt_id) + e)
            self.overlay_energy[tgt_id] = new_val

@dataclass
class Spore:
    """Переносимый пакет знаний для размножения."""
    resonators: List[dict] = field(default_factory=list)
    connections: List[Tuple[int, int, int]] = field(default_factory=list)
    parent_tick: int = 0
    
    def add_resonator(self, r: 'Resonator', label_id_map: Dict[str, int]):
        """Добавляет резонатор в спору."""
        if r.label not in label_id_map:
            label_id_map[r.label] = len(label_id_map)
        self.resonators.append({
            'label_id': label_id_map[r.label],
            'label': r.label,
            'hdc_vector': r.hdc_vector,
            'state': int(r.state),
            'energy': r.energy // 2,
        })
    
    def inject_into(self, lattice: 'CrystalLattice'):
        """Впрыскивает спору в новый кристалл."""
        id_mapping = {}
        for r_data in self.resonators:
            new_r = lattice.get_or_create(r_data['label'])
            new_r.hdc_vector = r_data['hdc_vector']
            new_r.state = TruthValue(r_data['state'])
            new_r.energy = r_data['energy']
            id_mapping[r_data['label_id']] = new_r.id
        for src_label_id, tgt_label_id, packed_edge in self.connections:
            if src_label_id in id_mapping and tgt_label_id in id_mapping:
                src_id = id_mapping[src_label_id]
                tgt_id = id_mapping[tgt_label_id]
                if src_id in lattice.resonators and tgt_id in lattice.resonators:
                    lattice.resonators[src_id].connections[tgt_id] = packed_edge

class CrystalPopulation:
    def __init__(self):
        self.crystals: List[CrystalLattice] = []
        self.active_index: int = 0
        self.mitosis_threshold: int = 500
    
    @property
    def active(self) -> CrystalLattice:
        if not self.crystals:
            self.spawn_initial()
        return self.crystals[self.active_index]
    
    def spawn_initial(self) -> CrystalLattice:
        crystal = CrystalLattice()
        self.crystals.append(crystal)
        self.active_index = 0
        return crystal
    
    def check_mitosis_trigger(self) -> bool:
        """Проверяет, перенасыщен ли активный кристалл для деления."""
        crystal = self.active
        if crystal.is_dormant:
            return False
        if len(crystal.resonators) >= self.mitosis_threshold:
            return True
        if hasattr(crystal, '_last_read_count') and crystal._last_read_count > 300:
            return True
        return False
    
    # 🆕 Приоритет 0.2: Удалён дубликат метода
    def check_population_dynamics(self) -> bool:
        """
        🧬 v7.1: Автоматический выбор стратегии размножения.
        Возвращает True, если произошло размножение.
        """
        # --- Стратегия 1: Митоз (перенасыщение одиночки) ---
        if len(self.crystals) == 1:
            if self.check_mitosis_trigger():
                self.perform_mitosis()
                return True
            return False
        
        # --- Стратегия 2: Конъюгация (обмен опытом) ---
        active_crystals = [
            (i, c) for i, c in enumerate(self.crystals)
            if not c.is_dormant
        ]
        if len(active_crystals) >= 2:
            best_pair = None
            best_similarity = 0.0
            for i in range(len(active_crystals)):
                for j in range(i + 1, len(active_crystals)):
                    idx1, c1 = active_crystals[i]
                    idx2, c2 = active_crystals[j]
                    roots1 = {r.label for r in c1.resonators.values() if r.label.startswith('root:') and r.energy > 50}
                    roots2 = {r.label for r in c2.resonators.values() if r.label.startswith('root:') and r.energy > 50}
                    if not roots1 or not roots2:
                        continue
                    union = roots1 | roots2
                    intersection = roots1 & roots2
                    jaccard = len(intersection) / len(union) if union else 0
                    genome_diff = bin(c1.genome ^ c2.genome).count('1')
                    if jaccard > 0.3 and genome_diff > 5:
                        if jaccard > best_similarity:
                            best_similarity = jaccard
                            best_pair = (idx1, idx2)
            if best_pair is not None:
                print(f"\n🔬 [ЭВОЛЮЦИЯ] Обнаружены совместимые кристаллы "
                      f"#{best_pair[0]+1} и #{best_pair[1]+1} "
                      f"(схожесть ядер: {best_similarity:.2f})")
                self.perform_conjugation(best_pair[0], best_pair[1])
                return True
        
        # --- Стратегия 3: Митоз активного (если он перенасыщен в популяции > 1) ---
        active = self.active
        if not active.is_dormant and len(active.resonators) >= self.mitosis_threshold:
            self.perform_mitosis()
            return True
        
        return False
    
    def perform_mitosis(self):
        """
        🧬 Митоз: Деление перенасыщенного кристалла.
        """
        parent = self.active
        print(f"\n🧬 [МИТОЗ] Перенасыщение кристалла #{self.active_index + 1} ({len(parent.resonators)} узлов). Деление...")
        spore = parent.extract_spore(top_k_attractors=3)
        print(f"   📦 Извлечено ядро: {len(spore.resonators)} узлов")
        child = CrystalLattice()
        child.genome = self._mutate_genome(parent.genome)
        spore.inject_into(child)
        print(f"   🌱 Создан дочерний кристалл: {len(child.resonators)} узлов")
        print(f"   🧬 Мутировавший геном: Decay={child.decay_percent}%, "
              f"Thresh={child.entropy_threshold:.2f}")
        parent.enter_dormancy()
        print(f"   💤 Родитель перешел в режим Проводника")
        self.crystals.append(child)
        self.active_index = len(self.crystals) - 1
        print(f"   🎯 Активный кристалл: #{self.active_index + 1} "
              f"из {len(self.crystals)}")
    
    def perform_conjugation(self, idx1: int, idx2: int):
        """
        🧬 v7.1: Конъюгация — скрещивание двух кристаллов.
        """
        if idx1 == idx2 or idx1 >= len(self.crystals) or idx2 >= len(self.crystals):
            return
        parent1 = self.crystals[idx1]
        parent2 = self.crystals[idx2]
        print(f"\n🧬 [КОНЪЮГАЦИЯ] Скрещивание кристаллов #{idx1+1} и #{idx2+1}...")
        spore1 = parent1.extract_spore(top_k_attractors=3)
        spore2 = parent2.extract_spore(top_k_attractors=3)
        print(f"   📦 Ядро #{idx1+1}: {len(spore1.resonators)} узлов")
        print(f"   📦 Ядро #{idx2+1}: {len(spore2.resonators)} узлов")
        child = CrystalLattice()
        child.genome = self._blend_genomes(parent1.genome, parent2.genome)
        spore1.inject_into(child)
        spore2.inject_into(child)
        print(f"   🌱 Новый кристалл создан: {len(child.resonators)} узлов")
        print(f"   🧬 Смешанный геном: Decay={child.decay_percent}%, "
            f"Thresh={child.entropy_threshold:.2f}")
        parent1.enter_dormancy()
        parent2.enter_dormancy()
        print(f"   💤 Оба родителя перешли в режим Проводника")
        self.crystals.append(child)
        self.active_index = len(self.crystals) - 1
        print(f"   🎯 Активный кристалл: #{self.active_index + 1} "
            f"из {len(self.crystals)}")
    
    def _blend_genomes(self, genome1: int, genome2: int) -> int:
        """Смешивает два генома: побитовое среднее + мутация."""
        import random
        blended = 0
        for shift in [GENE_DECAY_SHIFT, GENE_ENTROPY_THRESH,
                    GENE_MUTATION_RATE, GENE_MAX_DEPTH,
                    GENE_ANALOGY_TOL, GENE_PARADOX_PENALTY]:
            v1 = (genome1 >> shift) & 0xFF
            v2 = (genome2 >> shift) & 0xFF
            avg = (v1 + v2) // 2
            mutation = random.randint(-max(1, avg // 10), max(1, avg // 10))
            avg = max(1, min(255, avg + mutation))
            blended |= (avg << shift)
        return blended
    
    def _mutate_genome(self, parent_genome: int) -> int:
        import random
        mutation_mask = 0
        shifts = [GENE_DECAY_SHIFT, GENE_ENTROPY_THRESH, GENE_MAX_DEPTH]
        for shift in shifts:
            mutation = random.randint(0, 0xFF) & 0x0F
            mutation_mask |= (mutation << shift)
        return parent_genome ^ mutation_mask
    
    def recall_from_hive_mind(self, query_label: str) -> bool:
        """
        🧠 Механизм Коллективного Бессознательного (Hive Mind).
        """
        active = self.active
        best_dormant_crystal = None
        best_node = None
        best_score = -1
        for crystal in self.crystals:
            if crystal is active or not getattr(crystal, 'is_dormant', False):
                continue
            node = crystal.find_best_resonator_by_label(query_label)
            if not node:
                continue
            score = node.energy + len(node.connections) * 50
            if score > best_score:
                best_score = score
                best_node = node
                best_dormant_crystal = crystal
        if not best_dormant_crystal or not best_node or best_score < 100:
            return False
        print(f"\n   🧠 [КОЛЛЕКТИВНОЕ БЕССОЗНАТЕЛЬНОЕ] Обнаружено воспоминание в спящем кристалле!")
        print(f"      📂 Источник: Кристалл #{self.crystals.index(best_dormant_crystal) + 1} ({len(best_dormant_crystal.resonators)} узлов)")
        print(f"      🎯 Якорь: '{best_node.label}' (Энергия: {best_node.energy}, Связей: {len(best_node.connections)})")
        engram_ids = {best_node.id}
        sorted_neighbors = sorted(best_node.connections.items(), key=lambda item: unpack_edge(item[1])[0], reverse=True)
        for tgt_id, packed in sorted_neighbors[:7]:
            if tgt_id in best_dormant_crystal.resonators:
                engram_ids.add(tgt_id)
        injected_nodes = 0
        injected_edges = 0
        for src_id in engram_ids:
            src_r = best_dormant_crystal.resonators[src_id]
            if src_r.label.startswith(('EPOCH:', 'mod:', 'cluster:', 'mdl:', 'skill:')) or src_r.label == 'SELF':
                continue
            active_r = active.get_or_create(src_r.label)
            active_r.inject_energy(max(50, src_r.energy // 4), active.tick_count, cap=active.calibration.energy_cap)
            injected_nodes += 1
            for tgt_id, packed in src_r.connections.items():
                if tgt_id in engram_ids:
                    tgt_r = best_dormant_crystal.resonators[tgt_id]
                    if tgt_r.label.startswith(('EPOCH:', 'mod:', 'cluster:', 'mdl:', 'skill:')) or tgt_r.label == 'SELF':
                        continue
                    w, et = unpack_edge(packed)
                    active.connect(src_r.label, tgt_r.label, weight=max(10, w // 2), edge_type=et)
                    injected_edges += 1
        print(f"      💡 Извлечено воспоминание: {injected_nodes} узлов, {injected_edges} связей впрыснуто в сознание.")
        return True
    
    def get_status(self) -> str:
        lines = [f"🌌 ПОПУЛЯЦИЯ КРИСТАЛЛОВ: {len(self.crystals)} экземпляров"]
        for i, crystal in enumerate(self.crystals):
            marker = " ← АКТИВНЫЙ" if i == self.active_index else ""
            state = "💤 Проводник" if crystal.is_dormant else "🔥 Активен"
            lines.append(f"   #{i+1}: {state} | {len(crystal.resonators)} узлов | Такт {crystal.tick_count}{marker}")
        return "\n".join(lines)

def plan(self, goal_label: str, horizon: int = 10) -> List[str]:
    """Планировщик (Обратная трассировка от цели)."""
    goal_r = self.find_best_resonator_by_label(goal_label)
    if not goal_r:
        goal_r = self.get_or_create(goal_label)
    
    # 🆕 Приоритет 0.3: Удалён дубликат вызова
    path_ids = self.tick_backward(goal_r.id, max_depth=horizon)
    
    result = [f"🔮 [ПЛАН] Поиск пути к цели '{goal_label}' (Горизонт: {horizon}):"]
    if len(path_ids) <= 1:
        result.append("  ⚠️ Вывод: В текущей топологии нет причинных путей к цели.")
        result.append("  💡 Совет: Попробуйте 'вмешайся' или добавьте факты через 'ввод'.")
        return result
    path_labels = []
    for pid in path_ids:
        if pid in self.resonators:
            lbl = self.resonators[pid].label
            if lbl != 'SELF' and not lbl.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')):
                path_labels.append(lbl)
    result.append(f"  🎯 Найден маршрут (Длина: {len(path_labels)} узлов):")
    result.append(f"  🔗 {' <- '.join(path_labels)}")
    cost = 0
    for i in range(len(path_ids) - 1):
        src_id, tgt_id = path_ids[i], path_ids[i+1]
        if src_id in self.resonators and tgt_id in self.resonators[src_id].connections:
            packed = self.resonators[src_id].connections[tgt_id]
            w, et = unpack_edge(packed)
            if et not in (EDGE_CAUSE, EDGE_EFFECT, EDGE_COND):
                cost += 10
            else:
                cost += 1
    result.append(f"  ⏱️ Оценка стоимости (Cost): {cost} тактов.")
    return result

CrystalLattice.plan = plan