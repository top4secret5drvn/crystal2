#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Когнитивные модули для Вердикта
Содержит:
- BeliefRevisionEngine: Управление конфликтами и контекстами (TMS/AGM)
- ConceptFormationEngine: Автоматическое формирование понятий (MDL)
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import itertools

# Локальные dataclass-структуры используются как lightweight-протоколы.
# Основной интерпретатор работает с объектами по атрибутам, поэтому это
# убирает хрупкий импорт устаревшего основного модуля и разрывает циклические импорты.
@dataclass
class Reason:
    kind: str
    name: str = ""
    args: tuple = ()
    parents: List["Reason"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    confidence: float = 1.0

    def get_meta(self, key: str) -> Any:
        queue = [self]
        seen = set()
        while queue:
            cur = queue.pop(0)
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if key in cur.metadata:
                return cur.metadata[key]
            queue.extend(getattr(cur, "parents", []))
        return None


class KnowledgeBase:
    pass


@dataclass
class Condition:
    cond_type: str
    pred: str = ""
    args: Tuple[str, ...] = ()
    expected_value: int = 0
    expected_pattern: str = ""
    op: str = ""
    threshold: str = ""
    meta_key: str = ""


@dataclass
class Rule:
    name: str
    args: Tuple[str, ...]
    conditions: List[Condition]
    reason: Reason

@dataclass
class Belief:
    """Узел убеждения (Belief) для TMS (Truth Maintenance System)."""
    value: int
    reason: Reason
    confidence: float
    source: str
    context: str
    timestamp: float
    status: str  # "active", "defeated"


class BeliefRevisionEngine:
    """
    Модуль C: Belief Revision Engine + зачатки Context Engine.
    Отвечает за разрешение конфликтов, учет источников, уверенности и контекстов.
    """
    def __init__(self):
        self.beliefs: Dict[str, List[Belief]] = {}
        self.source_reliability: Dict[str, float] = {"default": 1.0, "неизвестно": 1.0}
        self.active_context: str = "global"
        
    def set_source_reliability(self, source: str, reliability: float):
        self.source_reliability[source] = max(0.0, min(1.0, reliability))
        
    def get_effective_weight(self, belief: Belief) -> float:
        """Эффективный вес = уверенность факта * надежность источника."""
        src_rel = self.source_reliability.get(belief.source, 1.0)
        return belief.confidence * src_rel

    def add_belief(self, key: str, value: int, reason: Reason):
        conf = reason.confidence
        source = str(reason.get_meta("источник") or "неизвестно")
        context = str(reason.get_meta("контекст") or self.active_context)
        
        src_rel_meta = reason.get_meta("надежность_источника")
        if src_rel_meta is not None:
            try: self.set_source_reliability(source, float(src_rel_meta))
            except (ValueError, TypeError): pass
        
        new_belief = Belief(
            value=value, reason=reason, confidence=conf,
            source=source, context=context,
            timestamp=time.time(), status="active"
        )
        
        if key not in self.beliefs:
            self.beliefs[key] = []
            
        self.beliefs[key].append(new_belief)
        self._resolve_conflicts(key)
        
    def _resolve_conflicts(self, key: str):
        """AGM/TMS стиль разрешения противоречий внутри одного контекста."""
        beliefs = self.beliefs[key]
        contexts = {}
        for b in beliefs:
            if b.context not in contexts: contexts[b.context] = []
            contexts[b.context].append(b)
            
        for ctx, ctx_beliefs in contexts.items():
            active_beliefs = [b for b in ctx_beliefs if b.status == "active"]
            if len(active_beliefs) <= 1: continue
                
            values = set(b.value for b in active_beliefs if b.value != 0)
            if len(values) > 1:
                active_beliefs.sort(key=lambda b: (self.get_effective_weight(b), b.timestamp), reverse=True)
                winner = active_beliefs[0]
                
                for b in active_beliefs[1:]:
                    if b.value != winner.value:
                        b.status = "defeated"
                        
    def query(self, key: str, context: str = None) -> Optional[Tuple[int, Reason]]:
        if key not in self.beliefs: return None
            
        ctx = context or self.active_context
        beliefs = self.beliefs[key]
        
        ctx_beliefs = [b for b in beliefs if b.context == ctx and b.status == "active"]
        if ctx_beliefs:
            best = max(ctx_beliefs, key=lambda b: (self.get_effective_weight(b), b.timestamp))
            return (best.value, best.reason)
            
        if ctx != "global":
            global_beliefs = [b for b in beliefs if b.context == "global" and b.status == "active"]
            if global_beliefs:
                best = max(global_beliefs, key=lambda b: (self.get_effective_weight(b), b.timestamp))
                return (best.value, best.reason)
                
        all_ctx = [b for b in beliefs if b.context == ctx]
        if all_ctx and all(b.status == "defeated" for b in all_ctx):
            conflict_reason = Reason("противоречие", f"неразрешимый_конфликт_{key}")
            return (0, conflict_reason)
            
        return None


class ConceptFormationEngine:
    """
    Модуль E: Concept Formation Engine.
    Автоматически обнаруживает повторяющиеся структуры и создает новые понятия.
    Использует принцип MDL (Minimum Description Length) для оценки полезности.
    """
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.discovered_concepts: List[Dict] = []
        
    def analyze_facts(self, min_support: int = 2, min_confidence: float = 0.7):
        """
        Главный метод: сканирует все факты и ищет частые комбинации.
        """
        print("\n=== АНАЛИЗ КОНЦЕПТУАЛЬНОЙ СТРУКТУРЫ ===")
        
        objects = defaultdict(list)
        
        # Проверяем, есть ли revision_engine (новая архитектура) или только facts (старая)
        if hasattr(self.kb, 'revision_engine') and self.kb.revision_engine.beliefs:
            for key, belief_list in self.kb.revision_engine.beliefs.items():
                for belief in belief_list:
                    if belief.status != "active" or belief.value != 1: continue
                    self._extract_predicate(key, objects)
        else:
            # Фоллбэк на старую архитектуру
            for key, (val, reason) in self.kb.facts.items():
                if val != 1: continue
                self._extract_predicate(key, objects)
        
        print(f"Найдено {len(objects)} уникальных объектов")
        
        patterns = self._find_frequent_patterns(objects, min_support)
        
        for pattern, support in patterns.items():
            self._evaluate_pattern(pattern, support, objects, min_confidence)
            
        self.discovered_concepts.sort(key=lambda c: c['compression_gain'], reverse=True)
        
        print(f"\nОбнаружено {len(self.discovered_concepts)} потенциальных концептов")
        for i, concept in enumerate(self.discovered_concepts[:5], 1):
            print(f"\n--- Концепт #{i} ---")
            print(f"Паттерн: {concept['pattern']}")
            print(f"Поддержка: {concept['support']} объектов")
            print(f"Сжатие: {concept['compression_gain']:.2f} бит")
            print(f"Предложение: {concept['proposal']}")
    
        print(f"Найдено {len(objects)} уникальных объектов")
        # DEBUG: Показать, что видит движок
        for obj, preds in objects.items():
            pred_names = sorted(set(p[0] for p in preds))
            print(f"  Объект '{obj}': предикаты {pred_names}")

    def _extract_predicate(self, key: str, objects: Dict):
        """Извлекает предикат и аргументы из ключа."""
        m = re.match(r"([a-zA-Zа-яА-Я0-9_]+)\(([^)]*)\)", key)
        if not m: return
        pred = m.group(1)
        args = tuple(a.strip() for a in m.group(2).split(","))
        if args:
            obj = args[0]
            objects[obj].append((pred, args[1:] if len(args) > 1 else ()))
    
    def _find_frequent_patterns(self, objects: Dict, min_support: int) -> Dict[Tuple, int]:
        """Ищет частые комбинации предикатов."""
        pattern_counts = defaultdict(int)
        
        for obj, predicates in objects.items():
            pred_names = tuple(sorted(set(p[0] for p in predicates)))
            for r in range(2, len(pred_names) + 1):
                for combo in itertools.combinations(pred_names, r):
                    pattern_counts[combo] += 1
        
        return {p: c for p, c in pattern_counts.items() if c >= min_support}
    
    def _evaluate_pattern(self, pattern: Tuple, support: int, objects: Dict, min_confidence: float):
        """Оценивает паттерн по принципу MDL."""
        objects_with_pattern = []
        for obj, predicates in objects.items():
            pred_names = set(p[0] for p in predicates)
            if all(p in pred_names for p in pattern):
                objects_with_pattern.append(obj)
        
        if len(objects_with_pattern) < support: return
        
        consequent_counts = defaultdict(int)
        for obj in objects_with_pattern:
            for pred, args in objects[obj]:
                if pred not in pattern:
                    consequent_counts[pred] += 1
        
        if not consequent_counts: return
        best_consequent = max(consequent_counts.items(), key=lambda x: x[1])
        consequent_pred, consequent_count = best_consequent
        
        confidence = consequent_count / len(objects_with_pattern)
        if confidence < min_confidence: return
        
        original_bits = support * len(pattern) + consequent_count
        compressed_bits = len(pattern) + 1 + consequent_count
        compression_gain = original_bits - compressed_bits
        
        if compression_gain <= 0: return
        
        pattern_str = " + ".join(pattern)
        proposal = f"правило обобщение_{hash(pattern) % 1000}(X) :- {', '.join(f'{p}(X) = да' for p in pattern)} -> {consequent_pred}(X) = да"
        
        self.discovered_concepts.append({
            'pattern': pattern,
            'support': support,
            'confidence': confidence,
            'consequent': consequent_pred,
            'compression_gain': compression_gain,
            'proposal': proposal,
            'objects': objects_with_pattern
        })
    
    def auto_generalize(self, top_n: int = 3):
        """Автоматически добавляет лучшие обнаруженные концепты как новые правила."""
        if not self.discovered_concepts:
            print("\nНет концептов для автоматического обобщения")
            return
        
        print(f"\n=== АВТОМАТИЧЕСКОЕ ОБОБЩЕНИЕ (Топ-{top_n}) ===")
        for concept in self.discovered_concepts[:top_n]:
            print(f"\nДобавляю правило: {concept['proposal']}")
            print(f"  Поддержка: {concept['support']}, Уверенность: {concept['confidence']:.2f}")
            
            rule_name = concept['consequent']
            args = ("X",)
            conditions = [
                Condition("fact", pred=p, args=("X",), expected_value=1)
                for p in concept['pattern']
            ]
            reason = Reason("правило", "mdl_concept", confidence=concept['confidence'])
            reason.metadata['auto_generated'] = True
            reason.metadata['support'] = concept['support']
            
            new_rule = Rule(rule_name, args, conditions, reason)
            self.kb.rules.append(new_rule)
            print(f"  ✓ Правило '{rule_name}' добавлено в базу знаний")
