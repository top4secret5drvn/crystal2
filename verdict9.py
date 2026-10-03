#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерпретатор языка «Троица» v0.9.1 (Cognitive Architecture)
ИСПРАВЛЕННАЯ ВЕРСИЯ:
- Устранено дублирование классов (импорт из cognitive_modules)
- Исправлен онтологический фоллбэк для наследования свойств
- Безопасный парсинг трит-значений
"""
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# Импортируем когнитивные модули (единственный источник правды для этих классов)
from cognitive_modules import BeliefRevisionEngine, ConceptFormationEngine

TRITS = {"да": 1, "нет": -1, "нез": 0}
TRIT_NAMES = {1: "да", -1: "нет", 0: "нез"}


@dataclass
class Reason:
    kind: str
    name: str = ""
    args: tuple = ()
    parents: List["Reason"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    confidence: float = 1.0

    def __str__(self):
        base = ""
        if self.kind == "факт": base = "факт"
        elif self.kind == "внешний": base = f"внешний({self.name})"
        elif self.kind == "правило": base = f"правило({self.name})"
        elif self.kind == "модуль":
            base = f"модуль({self.name}, {', '.join(self.args)})" if self.args else f"модуль({self.name})"
        elif self.kind == "онтология": base = f"онтология({self.name})"
        elif self.kind == "абдукция": base = f"абдукция({self.name})"
        elif self.kind == "опровержение": base = f"опровержение({self.name})"
        elif self.kind == "противоречие": base = f"противоречие({self.name})"
        elif self.kind == "синоним": base = f"синоним({self.name})"
        else: base = f"{self.kind}({self.name})"

        if self.metadata:
            meta_str = ", ".join(f"{k}:{v}" for k, v in self.metadata.items())
            return f"{base} [{meta_str}]"
        return base

    def get_meta(self, key: str) -> Any:
        queue = [self]
        seen = set()
        while queue:
            cur = queue.pop(0)
            if id(cur) in seen: continue
            seen.add(id(cur))
            if key in cur.metadata: return cur.metadata[key]
            queue.extend(cur.parents)
        return None

    def chain_length(self) -> int:
        nodes = set()
        self._collect_nodes(nodes)
        return len(nodes)

    def _collect_nodes(self, nodes: set):
        if id(self) in nodes: return
        nodes.add(id(self))
        for p in self.parents:
            p._collect_nodes(nodes)

    def _collect_leaves(self, facts, sources, seen=None):
        if seen is None: seen = set()
        if id(self) in seen: return
        seen.add(id(self))
        if not self.parents:
            if self.kind == "факт":
                facts.append(f"{self.name}({','.join(self.args)})" if self.args else self.name)
            elif self.kind == "внешний":
                sources.add(self.name)
                facts.append(f"внешний({self.name})")
            elif self.kind == "абдукция":
                facts.append(f"ГИПОТЕЗА: {self.name}({','.join(self.args)})" if self.args else f"ГИПОТЕЗА: {self.name}")
            elif self.kind == "противоречие":
                facts.append(f"КОНФЛИКТ: {self.name}")
            else:
                facts.append(str(self))
            src = self.metadata.get("источник")
            if src: sources.add(str(src))
        else:
            for p in self.parents:
                p._collect_leaves(facts, sources, seen)

    def generate_xai_report(self) -> str:
        facts = []
        sources = set()
        self._collect_leaves(facts, sources)
        complexity = self.chain_length()
        conf = self.confidence
        out_name = self.name if self.name else self.kind

        report = f"\n=== XAI ОТЧЕТ ===\n"
        report += f"ВЫВОД: {out_name} (Уверенность: {conf * 100:.0f}%)\n"
        report += f"Сложность рассуждения: {complexity} шагов. "
        if complexity <= 3:
            report += "Соответствует Бритве Оккама.\n"
        else:
            report += "ВНИМАНИЕ: Переусложненная цепочка! (Применен штраф)\n"
        if sources:
            report += f"Источники данных: {', '.join(sources)}\n"
        if facts:
            unique_facts = sorted(list(set(facts)))
            report += f"Базовые факты: {', '.join(unique_facts)}\n"
        report += "=================\n"
        return report

    def matches_pattern(self, pattern_str: str) -> bool:
        pattern_str = pattern_str.strip()
        queue = [self]
        seen = set()
        while queue:
            cur = queue.pop(0)
            if id(cur) in seen: continue
            seen.add(id(cur))
            if pattern_str == str(cur): return True
            m = re.match(r"([a-zA-Zа-яА-Я0-9_]+)\(([^)]*)\)", pattern_str)
            if m:
                if cur.kind == m.group(1) and cur.name == m.group(2).strip(): return True
            queue.extend(cur.parents)
        return False


@dataclass
class Demon:
    slot: str
    trigger_value: int
    action: str


@dataclass
class FrameTemplate:
    name: str
    slots: Dict[str, Tuple[int, Reason]] = field(default_factory=dict)
    demons: List[Demon] = field(default_factory=list)


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
class ModuleRule:
    condition_str: str
    output_port: str
    output_value: int
    reason: Reason


@dataclass
class FeedbackRule:
    condition_str: str
    memory_slot: str
    new_value: int
    reason: Reason


@dataclass
class Module:
    name: str
    inputs: List[str]
    outputs: List[str]
    memory: Dict[str, Tuple[int, Reason]]
    rules: List[ModuleRule]
    feedbacks: List[FeedbackRule]


class KnowledgeBase:
    """Единое определение базы знаний с интеграцией BeliefRevisionEngine."""

    def __init__(self):
        self.facts: Dict[str, Tuple[int, Reason]] = {}
        self.revision_engine = BeliefRevisionEngine()
        self.rules: List[Rule] = []
        self.frame_templates: Dict[str, FrameTemplate] = {}
        self.frame_instances: Dict[str, dict] = {}
        self.modules: Dict[str, Module] = {}

    def fact_key(self, pred, args): return f"{pred}({','.join(args)})"

    def set_fact(self, pred, args, value, reason):
        key = self.fact_key(pred, args)
        self.revision_engine.add_belief(key, value, reason)

        resolved = self.revision_engine.query(key)
        if resolved:
            val, res_reason = resolved
            self.facts[key] = (val, res_reason)
            if val == 0 and res_reason.kind == "противоречие":
                print(f"[РЕВИЗИЯ] Неразрешимый конфликт: {pred}{args} -> нез")
        else:
            self.facts[key] = (value, reason)

    def get_fact(self, pred, args):
        key = self.fact_key(pred, args)
        return self.revision_engine.query(key)


# === УНИФИКАЦИЯ ===
def is_variable(s): return bool(s) and s[0].isupper()


def parse_args(s):
    s = s.strip()
    return tuple(a.strip() for a in s.split(",")) if s else ()


def match_args(pattern, actual, env):
    if len(pattern) != len(actual): return None
    env = dict(env)
    for p, a in zip(pattern, actual):
        if is_variable(p):
            if p in env:
                if env[p] != a: return None
            else:
                env[p] = a
        else:
            if p != a: return None
    return env


def parse_metadata(meta_str: str) -> dict:
    meta = {}
    if not meta_str: return meta
    parts = re.split(r',\s*(?=[A-Za-zа-яА-Я0-9_]+\s*:)', meta_str)
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip()
            v = v.strip().strip('"\'')
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
            meta[k] = v
    return meta


# === ОНТОЛОГИЯ ===
def is_a(kb, child, parent, seen):
    if child == parent: return True
    if (child, parent) in seen: return False
    seen.add((child, parent))
    res = kb.get_fact("является", (child, parent))
    if res and res[0] == 1: return True
    for key, (val, reason) in kb.facts.items():
        if key.startswith("является(") and val == 1:
            m = re.match(r"является\(([^,]+),([^)]+)\)", key)
            if m:
                c, p = m.group(1).strip(), m.group(2).strip()
                if c == child:
                    if is_a(kb, p, parent, seen): return True
    return False


# === ВЫЧИСЛЕНИЕ ПРЕДИКАТА ===
def evaluate_predicate(kb, pred, args, env, seen_preds=None):
    if seen_preds is None:
        seen_preds = set()
    if pred in seen_preds:
        return None
    seen_preds.add(pred)

    resolved_args = tuple(env.get(a, a) for a in args)

    # Прямая проверка является(X, Y)
    if pred == "является":
        if len(resolved_args) == 2:
            child, parent = resolved_args
            if is_a(kb, child, parent, set()):
                return (1, Reason("онтология", f"наследование_{child}_{parent}"))
        return None

    # Прямой поиск факта
    direct = kb.get_fact(pred, resolved_args)
    if direct is not None: return direct

    # === ОНТОЛОГИЧЕСКИЙ ФОЛЛБЭК (ИСПРАВЛЕННАЯ ВЕРСИЯ) ===
    for key, (val, reason) in kb.facts.items():
        if key.startswith("является(") and val == 1:
            m = re.match(r"является\(([^,]+),([^)]+)\)", key)
            if not m: continue
            sub_class, super_class = m.group(1).strip(), m.group(2).strip()

            # Случай 1: Запрошен предикат-суперкласс
            if super_class == pred:
                sub_res = evaluate_predicate(kb, sub_class, resolved_args, env, seen_preds)
                if sub_res is not None and sub_res[0] == 1:
                    inh_reason = Reason("онтология", f"наследование_от_{sub_class}", parents=[sub_res[1]])
                    inh_reason.confidence = sub_res[1].confidence
                    return (1, inh_reason)

            # Случай 2: Аргумент запроса является суперклассом для известного подкласса
            if len(resolved_args) >= 1:
                arg_val = resolved_args[0]
                if arg_val == super_class or is_a(kb, super_class, arg_val, set()):
                    new_args = (sub_class,) + resolved_args[1:]
                    sub_res = evaluate_predicate(kb, pred, new_args, env, seen_preds)
                    if sub_res is not None and sub_res[0] == 1:
                        inh_reason = Reason("онтология", f"свойство_подкласса_{sub_class}", parents=[sub_res[1]])
                        inh_reason.confidence = sub_res[1].confidence
                        return (1, inh_reason)

    # Синонимический фоллбэк
    for key, (val, reason) in kb.facts.items():
        if key.startswith("синоним(") and val == 1:
            m = re.match(r"синоним\(([^,]+),([^)]+)\)", key)
            if m:
                w1, w2 = m.group(1).strip(), m.group(2).strip()
                if w1 == pred or w2 == pred:
                    target = w2 if w1 == pred else w1
                    sub_res = evaluate_predicate(kb, target, resolved_args, env, seen_preds)
                    if sub_res is not None and sub_res[0] == 1:
                        syn_reason = Reason("синоним", f"замена_{pred}_на_{target}", parents=[sub_res[1]])
                        syn_reason.confidence = sub_res[1].confidence * 0.99
                        return (1, syn_reason)

    # Правила
    for rule in kb.rules:
        if rule.name != pred: continue
        m = match_args(rule.args, resolved_args, {})
        if m is None: continue
        local_env = dict(m)
        all_ok = True
        collected_parents = []
        for cond in rule.conditions:
            ok, reason = check_condition(kb, cond, local_env)
            if not ok:
                all_ok = False
                break
            if reason is not None: collected_parents.append(reason)

        if all_ok:
            unique_parents = []
            seen_ids = set()
            for p in collected_parents:
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    unique_parents.append(p)

            rule_reason = Reason(
                rule.reason.kind, rule.reason.name, rule.reason.args,
                parents=unique_parents
            )
            parent_confs = [p.confidence for p in unique_parents if hasattr(p, 'confidence')]
            rule_conf = min(parent_confs) if parent_confs else 1.0

            base_conf = rule_reason.metadata.get("вес", 1.0)
            try:
                conf = rule_conf * float(base_conf)
            except:
                conf = rule_conf

            complexity = rule_reason.chain_length()
            occam_penalty = 0.90 ** max(0, complexity - 2)
            rule_reason.confidence = conf * occam_penalty

            return (1, rule_reason)
    return None


def check_condition(kb, cond, env):
    if cond.cond_type == "fact":
        resolved_args = tuple(env.get(a, a) for a in cond.args)
        res = evaluate_predicate(kb, cond.pred, resolved_args, env)
        if res is None: return (False, None)
        return (res[0] == cond.expected_value, res[1])
    elif cond.cond_type == "defeater":
        resolved_args = tuple(env.get(a, a) for a in cond.args)
        res = evaluate_predicate(kb, cond.pred, resolved_args, env)
        if res is not None and res[0] == cond.expected_value:
            return (False, Reason("опровержение", f"контрпример_{cond.pred}"))
        return (True, None)
    elif cond.cond_type == "tag_eq":
        resolved_args = tuple(env.get(a, a) for a in cond.args)
        res = evaluate_predicate(kb, cond.pred, resolved_args, env)
        if res is None: return (False, None)
        return (res[1].matches_pattern(cond.expected_pattern), res[1])
    elif cond.cond_type == "complexity":
        resolved_args = tuple(env.get(a, a) for a in cond.args)
        res = evaluate_predicate(kb, cond.pred, resolved_args, env)
        if res is None: return (False, None)
        complexity = res[1].chain_length()
        ok = False
        thresh = int(cond.threshold)
        if cond.op == "<=": ok = complexity <= thresh
        elif cond.op == ">=": ok = complexity >= thresh
        elif cond.op == "<": ok = complexity < thresh
        elif cond.op == ">": ok = complexity > thresh
        elif cond.op == "==": ok = complexity == thresh
        return (ok, res[1])
    elif cond.cond_type == "meta":
        resolved_args = tuple(env.get(a, a) for a in cond.args)
        res = evaluate_predicate(kb, cond.pred, resolved_args, env)
        if res is None: return (False, None)
        reason = res[1]
        val = reason.get_meta(cond.meta_key)
        if val is None: return (False, reason)
        ok = False
        try:
            val_num = float(val)
            thresh_num = float(cond.threshold)
            if cond.op == "<=": ok = val_num <= thresh_num
            elif cond.op == ">=": ok = val_num >= thresh_num
            elif cond.op == "<": ok = val_num < thresh_num
            elif cond.op == ">": ok = val_num > thresh_num
            elif cond.op == "==": ok = val_num == thresh_num
            elif cond.op == "!=": ok = val_num != thresh_num
        except (ValueError, TypeError):
            val_str = str(val).strip('"\'')
            thresh_str = str(cond.threshold).strip('"\'')
            if cond.op == "==": ok = val_str == thresh_str
            elif cond.op == "!=": ok = val_str != thresh_str
        return (ok, reason)
    return (False, None)


# === АБДУКЦИЯ ===
def abduce(kb, pred, args):
    for rule in kb.rules:
        if rule.name == pred:
            m = match_args(rule.args, args, {})
            if m is None: continue
            local_env = dict(m)
            for cond in rule.conditions:
                if cond.cond_type == "fact":
                    resolved_args = tuple(local_env.get(a, a) for a in cond.args)
                    res = evaluate_predicate(kb, cond.pred, resolved_args, local_env)
                    if res is None:
                        hyp_reason = Reason("абдукция", f"гипотеза_{cond.pred}", args=resolved_args, confidence=0.5)
                        kb.set_fact(cond.pred, resolved_args, 1, hyp_reason)
                        fire_demons(kb, cond.pred, resolved_args, 1)
                        print(f"[АБДУКЦИЯ] Выдвинута гипотеза: {cond.pred}({','.join(resolved_args)}) = да")


# === ПАРСЕР ===
def parse_value(s):
    s = s.strip()
    if s in TRITS:
        return TRITS[s]
    raise ValueError(f"Недопустимое трит-значение '{s}'. Парсер принимает только: да, нет, нез.")


def parse_reason(s):
    s = s.strip()
    if s == "факт": return Reason("факт")
    m = re.match(r"внешний\(([^)]+)\)", s)
    if m: return Reason("внешний", m.group(1).strip())
    m = re.match(r"правило\(([^)]+)\)", s)
    if m: return Reason("правило", m.group(1).strip())
    m = re.match(r"модуль\(([^,)]+)(?:,\s*([^)]+))?\)", s)
    if m:
        name = m.group(1).strip()
        args = tuple(a.strip() for a in m.group(2).split(",")) if m.group(2) else ()
        return Reason("модуль", name, args)
    return Reason("неизвестно", s)


def parse_file(text, kb):
    text = re.sub(r"//[^\n]*", "", text)
    blocks = []
    i = 0
    lines = text.split("\n")
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        if line.startswith("фрейм ") and "=" in line:
            buf = [line]
            if "{" in line and "}" not in line:
                i += 1
                while i < len(lines):
                    buf.append(lines[i])
                    if "}" in lines[i]: break
                    i += 1
            blocks.append(("\n".join(buf), "frame"))
        elif line.startswith("модуль ") and "{" in line:
            buf = [line]
            if "}" not in line:
                i += 1
                while i < len(lines):
                    buf.append(lines[i])
                    if "}" in lines[i]: break
                    i += 1
            blocks.append(("\n".join(buf), "module"))
        elif line.startswith("правило ") and ":-" in line:
            buf = [line]
            if "причина" not in line:
                i += 1
                while i < len(lines):
                    buf.append(lines[i])
                    if "причина" in lines[i]: break
                    i += 1
            blocks.append(("\n".join(buf), "rule"))
        else:
            blocks.append((line, "line"))
        i += 1

    for content, kind in blocks:
        if kind == "frame": parse_frame_block(content, kb)
        elif kind == "module": parse_module_block(content, kb)
        elif kind == "rule": parse_rule_block(content, kb)
        else: parse_single_line(content, kb)


def parse_frame_block(text, kb):
    m = re.match(r"фрейм\s+([A-Za-zа-яА-Я0-9_]+)\s*=\s*фрейм\(([^)]+)\)\s*\{([\s\S]*)\}", text)
    if not m: return
    name = m.group(1)
    body = m.group(3)
    tpl = FrameTemplate(name)
    for sm in re.finditer(r"слот\s+([A-Za-zа-яА-Я0-9_]+)\s*=\s*([^\.]+?)\s*причина\s+([^\.]+)\.", body):
        val = parse_value(sm.group(2).strip())
        if val is not None:
            tpl.slots[sm.group(1)] = (val, parse_reason(sm.group(3).strip()))
    for dm in re.finditer(r"демон\s+при_изменении\(([^)]+)\)\s*:\s*([^\-]+?)\s*->\s*([^\.]+)\.", body):
        val = parse_value(dm.group(2).strip())
        if val is not None:
            tpl.demons.append(Demon(dm.group(1).strip(), val, dm.group(3).strip()))
    kb.frame_templates[name] = tpl
    print(f"[ФРЕЙМ] Определён шаблон: {name}")


def parse_module_block(text, kb):
    m = re.match(r"модуль\s+([A-Za-zа-яА-Я0-9_]+)\s*\{([\s\S]*)\}", text)
    if not m: return
    name = m.group(1)
    body = m.group(2)
    inputs, outputs, memory, rules, feedbacks = [], [], {}, [], []
    im = re.search(r"входы\s*:\s*([^\.]+)\.", body)
    if im: inputs = [x.strip() for x in im.group(1).split(",")]
    om = re.search(r"выход\s*:\s*([^\.]+)\.", body)
    if om: outputs = [x.strip() for x in om.group(1).split(",")]
    for pm in re.finditer(r"память\s*:\s*([A-Za-zа-яА-Я0-9_]+)\s*=\s*([^\.]+?)\s*причина\s+([^\.]+)\.", body):
        val = parse_value(pm.group(2).strip())
        if val is not None:
            memory[pm.group(1)] = (val, parse_reason(pm.group(3).strip()))
    for rm in re.finditer(r"правило\s*:\s*([^\-]+?)\s*->\s*вых\(([^)]+)\)\s*=\s*([^\.]+?)\s*причина\s+([^\.]+)\.", body):
        val = parse_value(rm.group(3).strip())
        if val is not None:
            rules.append(ModuleRule(rm.group(1).strip(), rm.group(2).strip(), val, parse_reason(rm.group(4).strip())))
    for fm in re.finditer(r"обратная_связь\s*:\s*([^\-]+?)\s*->\s*пам\(([^)]+)\)\s*=\s*([^\.]+?)\s*причина\s+([^\.]+)\.", body):
        val = parse_value(fm.group(3).strip())
        if val is not None:
            feedbacks.append(FeedbackRule(fm.group(1).strip(), fm.group(2).strip(), val, parse_reason(fm.group(4).strip())))
    kb.modules[name] = Module(name, inputs, outputs, memory, rules, feedbacks)
    print(f"[МОДУЛЬ] Определён: {name}")


def parse_rule_block(text, kb):
    text = " ".join(text.split())
    m = re.match(r"правило\s+([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*:-\s*([\s\S]+?)\s*причина\s+([^\.]+)\.", text)
    if not m: return
    name = m.group(1)
    args = parse_args(m.group(2))
    conds_str = m.group(3).strip()
    reason = parse_reason(m.group(4).strip())
    conditions = []
    pos = 0
    while pos < len(conds_str):
        while pos < len(conds_str) and conds_str[pos] in " ,\n\t": pos += 1
        if pos >= len(conds_str): break
        remaining = conds_str[pos:]

        m_meta = re.match(r"мета\(\s*([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*,\s*([A-Za-zа-яА-Я0-9_]+)\s*\)\s*([<>=!]+)\s*([^\s,]+)", remaining)
        if m_meta:
            conditions.append(Condition("meta", pred=m_meta.group(1), args=parse_args(m_meta.group(2)), meta_key=m_meta.group(3), op=m_meta.group(4), threshold=m_meta.group(5)))
            pos += m_meta.end(); continue

        m_def = re.match(r"исключение\(\s*([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*=\s*([^\s,]+)\s*\)", remaining)
        if m_def:
            val = parse_value(m_def.group(3).strip())
            if val is not None:
                conditions.append(Condition("defeater", pred=m_def.group(1), args=parse_args(m_def.group(2)), expected_value=val))
            pos += m_def.end(); continue

        m1 = re.match(r"сложность\(\s*тег\(\s*([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*\)\s*\)\s*([<>=!]+)\s*(\d+)", remaining)
        if m1:
            conditions.append(Condition("complexity", pred=m1.group(1), args=parse_args(m1.group(2)), op=m1.group(3), threshold=m1.group(4)))
            pos += m1.end(); continue

        m2 = re.match(r"тег\(\s*([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*\)\s*=\s*([^\s,]+(?:\([^)]*\))?)", remaining)
        if m2:
            conditions.append(Condition("tag_eq", pred=m2.group(1), args=parse_args(m2.group(2)), expected_pattern=m2.group(3).strip()))
            pos += m2.end(); continue

        m3 = re.match(r"([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*=\s*([^\s,]+)", remaining)
        if m3:
            val = parse_value(m3.group(3).strip())
            if val is not None:
                conditions.append(Condition("fact", pred=m3.group(1), args=parse_args(m3.group(2)), expected_value=val))
            pos += m3.end(); continue

        comma = remaining.find(",")
        if comma == -1: break
        pos += comma + 1
    kb.rules.append(Rule(name, args, conditions, reason))


def parse_single_line(line, kb):
    line = line.strip()
    if not line: return

    m = re.match(r"факт\s+([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*=\s*([^\.]+?)\s*причина\s+([^\. \[]+)(?:\s*\[([^\]]+)\])?\.", line)
    if m:
        pred = m.group(1)
        args = parse_args(m.group(2))
        val = parse_value(m.group(3).strip())
        if val is None:
            return
        reason = parse_reason(m.group(4).strip())
        if m.group(5):
            reason.metadata = parse_metadata(m.group(5))
            if "надежность" in reason.metadata:
                try: reason.confidence = float(reason.metadata["надежность"])
                except: pass
            elif "confidence" in reason.metadata:
                try: reason.confidence = float(reason.metadata["confidence"])
                except: pass
        kb.set_fact(pred, args, val, reason)
        print(f"[ФАКТ] {pred}({','.join(args)}) = {TRIT_NAMES[val]}, причина: {reason} (Уверенность: {reason.confidence:.2f})")
        fire_demons(kb, pred, args, val)
        return

    m = re.match(r"факт\s+([A-Za-zа-яА-Я0-9_]+)\s*=\s*фрейм\(([^)]+)\)\.", line)
    if m:
        inst_name = m.group(1)
        tpl_name = m.group(2).strip()
        if tpl_name not in kb.frame_templates:
            print(f"[ОШИБКА] Шаблон фрейма {tpl_name} не найден"); return
        tpl = kb.frame_templates[tpl_name]
        kb.frame_instances[inst_name] = {}
        for slot, (val, reason) in tpl.slots.items():
            kb.set_fact(slot, (inst_name,), val, reason)
        kb.set_fact(inst_name, (), 1, Reason("факт"))
        print(f"[ФРЕЙМ] Создан: {inst_name} (шаблон: {tpl_name})")
        return

    m = re.match(r"связь\s*:\s*([A-Za-zа-яА-Я0-9_]+)\s*->\s*([A-Za-zа-яА-Я0-9_]+)\.([A-Za-zа-яА-Я0-9_]+)\.", line)
    if m:
        src, mod_name, port = m.group(1), m.group(2), m.group(3)
        if mod_name in kb.modules:
            mod = kb.modules[mod_name]
            mod.memory[port] = (1, Reason("факт"))
            kb.set_fact(f"{mod_name}_{port}", (src,), 1, Reason("факт"))
            print(f"[СВЯЗЬ] {src} -> {mod_name}.{port}")
            execute_module(kb, mod)
        return

    m = re.match(r"цель\s*:\s*([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*=\s*([A-Za-zа-яА-Я0-9_]+)\s+причина\s+([A-Za-zа-яА-Я0-9_]+)\.", line)
    if m:
        pred = m.group(1)
        args = parse_args(m.group(2))
        var_val = m.group(3)
        res = evaluate_predicate(kb, pred, args, {})
        if res is None:
            abduce(kb, pred, args)
            res = evaluate_predicate(kb, pred, args, {})
        if res is None:
            print(f"  -> {var_val} = нез (не выведено)")
        else:
            print(f"  -> {var_val} = {TRIT_NAMES[res[0]]}, причина = {res[1]}")
            if res[0] == 1:
                print(res[1].generate_xai_report())
        return

    m = re.match(r"цель\s*:\s*память\s+([A-Za-zа-яА-Я0-9_]+)\.([A-Za-zа-яА-Я0-9_]+)\s*=\s*([A-Za-zа-яА-Я0-9_]+)\s+причина\s+([A-Za-zа-яА-Я0-9_]+)\.", line)
    if m:
        mod_name, slot, var_val = m.group(1), m.group(2), m.group(3)
        if mod_name in kb.modules and slot in kb.modules[mod_name].memory:
            val, rsn = kb.modules[mod_name].memory[slot]
            print(f"  -> {var_val} = {TRIT_NAMES[val]}, причина = {rsn}")
        else: print(f"  -> {var_val} = нез (нет в памяти)")
        return


def fire_demons(kb, pred, args, value):
    if not args: return
    inst_name = args[0]
    if inst_name not in kb.frame_instances: return
    for tpl_name, tpl in kb.frame_templates.items():
        for demon in tpl.demons:
            if demon.slot == pred and demon.trigger_value == value:
                print(f"[ДЕМОН {tpl_name}] Сработал триггер на {pred}: {demon.action}")
                pm = re.match(r'печать\(\s*"([^"]+)"\s*\)', demon.action)
                if pm: print(pm.group(1))


# === МОДУЛИ ===
def execute_module(kb, mod):
    outputs = {}
    for rule in mod.rules:
        if eval_module_condition(kb, mod, rule.condition_str):
            outputs[rule.output_port] = (rule.output_value, rule.reason)
            print(f"[МОДУЛЬ {mod.name}] вых({rule.output_port}) = {TRIT_NAMES[rule.output_value]}, причина: {rule.reason}")
    for port, (val, rsn) in outputs.items():
        mod.memory[f"_out_{port}"] = (val, rsn)
    for fb in mod.feedbacks:
        if eval_feedback_condition(kb, mod, outputs, fb.condition_str):
            mod.memory[fb.memory_slot] = (fb.new_value, fb.reason)
            print(f"[ГОМЕОСТАЗИС {mod.name}] пам({fb.memory_slot}) -> {TRIT_NAMES[fb.new_value]}, причина: {fb.reason}")


def eval_module_condition(kb, mod, cond):
    parts = [p.strip() for p in cond.split("&")]
    for part in parts:
        m = re.match(r"(вх|пам)\(([^)]+)\)\s*=\s*([^\s]+)", part)
        if not m: return False
        kind, slot, expected = m.group(1), m.group(2).strip(), m.group(3).strip()
        if kind == "вх":
            if not any(key.startswith(f"{mod.name}_{slot}(") and key.endswith(f"({expected})") for key in kb.facts): return False
        elif kind == "пам":
            if slot not in mod.memory: return False
            val, _ = mod.memory[slot]
            if expected in TRITS:
                if val != TRITS[expected]: return False
            else: return False
    return True


def eval_feedback_condition(kb, mod, outputs, cond):
    cond = cond.strip()
    m_out = re.match(r"сложность\(\s*тег\(\s*вых\(([^)]+)\)\s*\)\s*\)\s*([<>=!]+)\s*(\d+)", cond)
    if m_out:
        port, op, threshold = m_out.group(1).strip(), m_out.group(2), int(m_out.group(3))
        if port in outputs: _, rsn = outputs[port]
        else:
            key = f"_out_{port}"
            if key not in mod.memory: return False
            _, rsn = mod.memory[key]
        complexity = rsn.chain_length()
    else:
        m_pred = re.match(r"сложность\(\s*тег\(\s*([A-Za-zа-яА-Я0-9_]+)\(([^)]*)\)\s*\)\s*\)\s*([<>=!]+)\s*(\d+)", cond)
        if m_pred:
            pred = m_pred.group(1)
            args_str = m_pred.group(2)
            op = m_pred.group(3)
            threshold = int(m_pred.group(4))
            args = tuple(a.strip() for a in args_str.split(",")) if args_str else ()
            res = evaluate_predicate(kb, pred, args, {})
            if res is None: return False
            complexity = res[1].chain_length()
        else: return False

    if op == ">": return complexity > threshold
    if op == ">=": return complexity >= threshold
    if op == "<": return complexity < threshold
    if op == "<=": return complexity <= threshold
    if op == "==": return complexity == threshold
    return False


# === ГЛАВНАЯ ===
def run(filename=None):
    kb = KnowledgeBase()
    if filename is None:
        text = BUILTIN_DEMO
        print("=== Запуск встроенного демо (v0.9.1 Cognitive Modules) ===\n")
    else:
        with open(filename, "r", encoding="utf-8") as f: text = f.read()
        print(f"=== Запуск файла: {filename} ===\n")
    parse_file(text, kb)

    concept_engine = ConceptFormationEngine(kb)
    concept_engine.analyze_facts(min_support=2, min_confidence=0.5)
    concept_engine.auto_generalize(top_n=2)

    print("\n=== ПОВТОРНЫЙ ВЫВОД С НОВЫМИ ПРАВИЛАМИ ===")
    for line in text.split("\n"):
        if line.strip().startswith("цель:"):
            parse_single_line(line.strip(), kb)


BUILTIN_DEMO = """
// НОВОЕ (v0.8): Ревизия убеждений и Контексты
факт болен(иван) = да причина внешний(врач1) [надежность:0.9, источник:API_клиники, контекст:осмотр_12_05].
факт болен(иван) = нет причина внешний(врач2) [надежность:0.4, источник:опрос_соседа, контекст:осмотр_12_05].

факт болен(иван) = да причина факт [контекст:гипотеза_А].
факт болен(иван) = нет причина факт [контекст:гипотеза_Б].
"""

if __name__ == "__main__":
    if len(sys.argv) > 1: run(sys.argv[1])
    else: run()