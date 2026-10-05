"""
membrane.py — Переводчик между текстом и битами (v5.1 Cognitive Rewrite + Verdict Integration).
Вектор 3 (Морфология) + Вектор 5 (Голос) + 🧠 Раздел 40 (Когнитивный Конвейер)
+ 🆕 Приоритет 2: Обработка отрицания, вопросов, извлечение триплетов.
"""
import json
import os
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from engine import (TruthValue, EDGE_CAUSE, EDGE_EXCEPT, EDGE_COND, EDGE_SYNTAGM,
                    EDGE_IS_A, EDGE_PART_OF, EDGE_GOAL, EDGE_ANALOG, unpack_edge, pack_edge, MarkerType, Marker, DependencyNode, CrystalReason,
                    NodeType)  # 🆕 Фаза 1: типология узлов + SYN-слой; 🆕 Фаза 2: EDGE_GOAL/EDGE_ANALOG (операторы 'для'/'или')
from calibration import CalibrationProfile


@dataclass
class CausalStats:
    """Статистика извлеченной логики из текста."""
    cause_links: int = 0
    except_links: int = 0
    cond_links: int = 0
    syntagm_links: int = 0
    is_a_links: int = 0
    negations: int = 0
    questions: int = 0
    bind_links: int = 0      # 🆕 Фаза 2: применено операторное объединение 'и'


class SuffixTrie:
    """Суффиксное дерево для мгновенного O(L) поиска подстрок и корней."""
    def __init__(self):
        self.root: Dict = {}
        self.word_to_id: Dict[bytes, int] = {}
        self.id_to_word: Dict[int, bytes] = {}
        self._next_id = 0

    def insert(self, word: bytes, word_id: Optional[int] = None) -> int:
        if word_id is None:
            word_id = self._next_id
            self._next_id += 1
        node = self.root
        for byte in word:
            if byte not in node:
                node[byte] = {}
            node = node[byte]
        node['$'] = word_id
        self.word_to_id[word] = word_id
        self.id_to_word[word_id] = word
        return word_id

    def find_lcs(self, words: List[bytes]) -> bytes:
        if not words:
            return b''
        if len(words) == 1:
            return words[0]
        words = sorted(words, key=len)
        shortest = words[0]
        for length in range(len(shortest), 2, -1):
            for start in range(len(shortest) - length + 1):
                candidate = shortest[start:start + length]
                if all(candidate in w for w in words[1:]):
                    return candidate
        return b''

    def find_nearest(self, hdc_vector: int, lattice, top_k: int = 5) -> List[Tuple[str, float]]:
        candidates = []
        for label, r_id in lattice.label_to_id.items():
            if label.startswith(('root:', 'mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')) or label == 'SELF':
                continue
            r = lattice.resonators[r_id]
            sim = lattice.encoder.similarity(r.hdc_vector, hdc_vector)
            candidates.append((label, sim))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


class RussianStemmer:
    """Упрощённый стеммер Портера для русского. Без зависимостей."""

    PERFECTIVE_GERUND = ('ив', 'ивши', 'ившись', 'ыв', 'ывши', 'ывшись')
    REFLEXIVE = ('ся', 'сь')
    ADJECTIVE = (
        'ее', 'ие', 'ые', 'ое', 'ими', 'ыми', 'ей', 'ий', 'ый', 'ой',
        'ем', 'им', 'ым', 'ом', 'его', 'ого', 'ему', 'ому', 'их', 'ых',
        'ую', 'юю', 'ая', 'яя', 'ою', 'ею',
    )
    PARTICIPLE = ('ем', 'нн', 'вш', 'ющ', 'щ')
    VERB = (
        'ила', 'ыла', 'ена', 'ейте', 'уйте', 'ите', 'или', 'ыли', 'ей',
        'уй', 'ил', 'ыл', 'им', 'ым', 'ен', 'ило', 'ыло', 'ено', 'ят',
        'ует', 'уют', 'ит', 'ыт', 'ены', 'ить', 'ыть', 'ишь', 'ую', 'ю',
        'ать', 'еть', 'оть', 'уть', 'ть',
    )
    NOUN = (
        'а', 'ев', 'ов', 'ие', 'ье', 'е', 'иями', 'ями', 'ами', 'еи',
        'ии', 'и', 'ией', 'ей', 'ой', 'ий', 'й', 'иям', 'ям', 'ием',
        'ем', 'ам', 'ом', 'о', 'у', 'ах', 'иях', 'ях', 'ы', 'ь', 'ию',
        'ью', 'ю', 'ия', 'ья', 'я',
    )
    SUPERLATIVE = ('ейш', 'ейше')
    DERIVATIONAL = ('ост', 'ость')

    def stem(self, word: str) -> str:
        word = word.lower().strip()
        if len(word) <= 3:
            return word

        # Шаг 1: Найти окончание (окончание = последняя гласная + всё после)
        rv_region = self._find_rv(word)
        if not rv_region:
            return word

        stem = word[:len(word) - len(rv_region)]
        ending = rv_region

        # Шаг 2: Удалить совершенный герундий
        for suffix in sorted(self.PERFECTIVE_GERUND, key=len, reverse=True):
            if ending.endswith(suffix):
                ending = ending[:-len(suffix)]
                break
        else:
            # Шаг 3: Удалить возвратное
            for suffix in self.REFLEXIVE:
                if ending.endswith(suffix):
                    ending = ending[:-len(suffix)]
                    break
            # Шаг 4: Удалить прилагательное, причастие или глагол
            done = False
            for group in (self.ADJECTIVE, self.PARTICIPLE, self.VERB):
                for suffix in sorted(group, key=len, reverse=True):
                    if ending.endswith(suffix):
                        ending = ending[:-len(suffix)]
                        done = True
                        break
                if done:
                    break
            if not done:
                # Шаг 5: Удалить существительное
                for suffix in sorted(self.NOUN, key=len, reverse=True):
                    if ending.endswith(suffix):
                        ending = ending[:-len(suffix)]
                        break

        result = stem + ending

        # Шаг 6: Удалить превосходную степень
        for suffix in self.SUPERLATIVE:
            if result.endswith(suffix):
                result = result[:-len(suffix)]
                break

        # Шаг 7: Удалить деривационный суффикс
        for suffix in self.DERIVATIONAL:
            if result.endswith(suffix) and len(result) - len(suffix) >= 3:
                result = result[:-len(suffix)]
                break

        # Шаг 8: Удалить конечный мягкий знак
        if result.endswith('ь') and len(result) > 3:
            result = result[:-1]

        # 🆕 Фикс: минимальная длина стема — 4 символа для существительных
        # (предотвращает "яблоко" → "яб", "цитрус" → "ци")
        if len(result) < 4:
            return word
        return result

    def _find_rv(self, word: str) -> str:
        """Найти RV-регион: всё после ПЕРВОЙ гласной."""
        vowels = set('аеиоуыэюяё')
        found_first = False
        for i, ch in enumerate(word):
            if ch in vowels:
                if not found_first:
                    found_first = True
                    continue
                # Нашли вторую гласную — RV начинается после неё
                return word[i + 1:]
        # Если гласная только одна — возвращаем всё после неё
        for i, ch in enumerate(word):
            if ch in vowels:
                return word[i + 1:]
        return ''


class LanguageMembrane:
    """
    Сенсорная мембрана: единственная точка входа текста в Кристалл.
    """
    # 🆕 ФАЗА 2, Шаг 2.1 (идея #7): Реестр служебных слов-операторов.
    # «и», «не», «если», «но» — НЕ концепты. Это ОПЕРАЦИИ над графом.
    # Ключ: словоформа; значение: тип операции.
    OPERATORS = {
        'и':      'bind',        # объединение (суперпозиция в working_memory)
        'или':    'branch',      # развилка
        'не':     'negate',      # отрицание (уже частично было)
        'но':     'contrast',    # конфликт/контраст → EDGE_EXCEPT
        'если':   'condition',   # условие → EDGE_COND
        'то':     'consequence', # следствие
        'в':      'container',   # пространственное отношение
        'на':     'surface',
        'для':    'purpose',     # целевая направленность → EDGE_GOAL
        'из-за':  'cause',       # причинная связь → EDGE_CAUSE
        'потому': 'cause',
        'поэтому': 'consequence',
        'это':    'is_a',        # связка (уже была!) → EDGE_IS_A
        'является': 'is_a',
        'часть':  'part_of',     # → EDGE_PART_OF
        'из':     'part_of',
    }

    # 🛡 Жесткий стоп-лист — только РЕАЛЬНО пустые слова.
    # ⚠️ ФАЗА 2: 'не', 'и', 'но', 'если', 'для', 'из-за', 'это', 'является',
    # 'часть', 'или', 'то', 'потому', 'поэтому' УБРАНЫ — они операторы (OPERATORS),
    # а не мусор. Операторы перехватываются раньше стоп-листа.
    STOP_WORDS = {
        'а', 'с', 'к', 'о', 'у', 'от', 'по',
        'за', 'до', 'он', 'она', 'оно', 'мы', 'вы', 'я', 'же', 'ли',
        'бы', 'что', 'как', 'вот', 'еще', 'ещё', 'уже',
        'да', 'нет', 'был', 'была', 'было', 'были', 'будет', 'есть',
        'ни', 'её', 'его', 'их', 'мой', 'твой', 'наш', 'ваш', 'свой',
        'этот', 'тот', 'такой', 'там', 'тут', 'где', 'когда', 'без',
        'под', 'над', 'про', 'при', 'через', 'между', 'лишь', 'только',
        'том', 'тому', 'тем', 'тех', 'того', 'та', 'те', 'весь', 'вся', 'всё',
        'все', 'всего', 'всей', 'всем', 'всех', 'сам', 'сама', 'само', 'сами',
        'который', 'которая', 'которое', 'которые', 'именно',
        'этом', 'этой', 'этих', 'кто', 'куда', 'откуда', 'почему', 'зачем',
        'так', 'этак', 'иначе', 'затем', 'потом', 'сначала', 'наконец',
        'впрочем', 'ведь', 'хоть', 'хотя', 'пускай', 'пусть', 'будто',
        'словно', 'точно', 'якобы', 'вроде', 'типа', 'даже', 'просто',
        'почти', 'опять', 'снова', 'всегда', 'никогда', 'иногда', 'часто',
        'редко', 'обычно', 'вероятно', 'возможно', 'конечно', 'безусловно',
        'разумеется', 'действительно', 'пожалуй', 'наверное', 'неужели', 'разве'
    }

    CONSTRUCTIONS = {
        "CAUSATION": "{Cause} приводит к {Effect}.",
        "STATE_ADJ": "{Entity} — {Property}.",
        "STATE_NOUN": "{Entity} — это {Property}.",
        "IS_A": "{Entity} является {Class}.",
        "PART_OF": "{Part} входит в состав {Whole}.",
        "COMPARISON": "{Entity1} и {Entity2} имеют общие черты: {Common}.",
        "NEGATION": "{Entity} не является {Class}.",
    }

    CAUSE_MARKERS = frozenset([
        'потому', 'поэтому', 'следовательно', 'значит', 'так',
        'из-за', 'благодаря', 'вызывает', 'приводит', 'ведет',
        'обусловлено', 'результат', 'причина'
    ])
    EXCEPT_MARKERS = frozenset([
        'но', 'однако', 'кроме', 'исключение', 'вопреки',
        'несмотря', 'хотя', 'зато', 'иначе'
    ])
    COND_MARKERS = frozenset([
        'если', 'когда', 'условие', 'случай', 'при', 'допустим'
    ])
    IS_A_MARKERS = frozenset(['является', 'есть', 'представляет'])  # 🆕 Приоритет 2.3 ('это' — теперь оператор, не маркер-концепт)
    # 🆕 Приоритет 2.2: Детекция намерений вопросов
    QUESTION_INTENTS = {
        'why': frozenset(['почему', 'зачем', 'отчего']),
        'what': frozenset(['что', 'кто', 'какой', 'какая', 'какое', 'какие']),
        'how': frozenset(['как', 'каким', 'образом']),
        'compare': frozenset(['сравни', 'отличие', 'разница', 'отличи']),
    }

    def __init__(self, lattice):
        self.lattice = lattice
        self.calibration = lattice.calibration
        self.trie = SuffixTrie()
        self.known_words: Set[bytes] = set()
        self.word_forms: Dict[bytes, List[bytes]] = {}
        self.learned_rules: Dict[str, int] = {}
        self.contextual_neighbors: Dict[str, Dict[str, float]] = {}
        self.word_occurrence_count: Dict[str, int] = {}
        self.trigram_neighbors: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.learned_rules_path = os.path.join(os.getcwd(), 'learned_language_rules.json')
        self._label_cache: Dict[str, str] = {}
        self.stemmer = RussianStemmer()
        # 🆕 ФАЗА 2 (Шаг 2.2): состояние отложенных операторов + карта
        # "концепт → последний токен" для SYN-ребра BIND (оператор 'и')
        self._last_token_by_label: Dict[str, int] = {}
        self._pending_negate = False
        self._pending_condition = False
        self._pending_cause = False
        self._pending_is_a = False
        self._pending_goal = False
        # 🆕 Фаза 2 (двухпроходные операторы): левый член пары 'A и B' / 'A или B'
        self._pending_bind: str = ''
        self._pending_branch: str = ''
        self._goal_target_label: Optional[str] = None
        self.last_query_intent: Optional[str] = None  # 🆕 Приоритет 2.2
        self._load_learned_rules()

    def invalidate_cache(self):
        old_size = len(self._label_cache)
        self._label_cache.clear()
        if old_size > 0:
            print(f"   🔄 [МЕМБРАНА] Кэш морфологии сброшен ({old_size} записей)")

    def _get_surface_form(self, label: str) -> str:
        if not label.startswith("root:"):
            return label
        root_bytes = label[5:].encode('utf-8')
        best_word = label[5:]
        max_count = -1
        for w_bytes, count in self.word_occurrence_count.items():
            if isinstance(w_bytes, bytes):
                if root_bytes in w_bytes and count > max_count:
                    max_count = count
                    best_word = w_bytes.decode('utf-8', errors='ignore')
        return best_word

    def _learn_contextual_rules(self, words: List[str], resolved_labels: List[Optional[str]]):
        labels = [lbl for lbl in resolved_labels if lbl]
        if len(labels) < 2:
            return
        for label, neighbors in list(self.contextual_neighbors.items()):
            for other in list(neighbors.keys()):
                neighbors[other] = max(0.3, neighbors[other] * 0.95)
                if neighbors[other] < 0.4:
                    del neighbors[other]
            if not neighbors:
                del self.contextual_neighbors[label]
            elif len(neighbors) > 6:
                ranked = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[:6]
                self.contextual_neighbors[label] = {k: v for k, v in ranked}
        window_size = 3
        for idx, label in enumerate(labels):
            window = labels[max(0, idx - window_size): min(len(labels), idx + window_size + 1)]
            for other in window:
                if other == label:
                    continue
                bucket = self.contextual_neighbors.setdefault(label, {})
                bucket[other] = bucket.get(other, 0.0) + 1.0
                bucket_other = self.contextual_neighbors.setdefault(other, {})
                bucket_other[label] = bucket_other.get(label, 0.0) + 1.0
        for i in range(len(labels) - 2):
            a, b, c = labels[i], labels[i+1], labels[i+2]
            if a and b and c:
                key = (a, b)
                self.trigram_neighbors.setdefault(key, {})
                self.trigram_neighbors[key][c] = self.trigram_neighbors[key].get(c, 0.0) + 1.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                if a == b:
                    continue
                if abs(i - j) <= 2:
                    score = self.contextual_neighbors.get(a, {}).get(b, 0.0)
                    if score >= 1.2:
                        weight = min(50, 12 + int(score * 10))
                        self.lattice.connect(a, b, weight=weight, edge_type=EDGE_SYNTAGM)
        self._persist_learned_rules()

    def _persist_learned_rules(self):
        """🆕 Приоритет 2.4: Сохранение выученных правил в JSON."""
        try:
            data = {
                'contextual_neighbors': {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in self.contextual_neighbors.items()
                },
                'trigram_neighbors': {
                    f"{k[0]}|||{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                    for k, v in self.trigram_neighbors.items()
                },
                'word_occurrence_count': {
                    (k.decode('utf-8', errors='ignore') if isinstance(k, bytes) else k): int(v)
                    for k, v in self.word_occurrence_count.items()
                },
            }
            with open(self.learned_rules_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"   ⚠️ Не удалось сохранить правила: {e}")

    def _load_learned_rules(self):
        """🆕 Приоритет 2.4: Загрузка выученных правил из JSON."""
        if not os.path.exists(self.learned_rules_path):
            return
        try:
            with open(self.learned_rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.contextual_neighbors = {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in data.get('contextual_neighbors', {}).items()
            }
            self.trigram_neighbors = {}
            for key_str, v in data.get('trigram_neighbors', {}).items():
                parts = key_str.split('|||', 1)
                if len(parts) == 2:
                    self.trigram_neighbors[(parts[0], parts[1])] = {
                        kk: float(vv) for kk, vv in v.items()
                    }
            self.word_occurrence_count = {}
            for k, v in data.get('word_occurrence_count', {}).items():
                try:
                    self.word_occurrence_count[k.encode('utf-8')] = int(v)
                except Exception:
                    pass
            loaded = len(self.contextual_neighbors) + len(self.trigram_neighbors)
            if loaded > 0:
                print(f"   📚 [МЕМБРАНА] Загружено правил: {loaded}")
        except Exception as e:
            print(f"   ⚠️ Не удалось загрузить правила: {e}")

    def resolve_query(self, word_str: str) -> str:
        cleaned = word_str.strip().lower()
        stem = self.stemmer.stem(cleaned)
        if stem != cleaned and len(stem) >= self.calibration.min_root_len:
            return f"root:{stem}"
        return cleaned

    # ================================================================
    # 🔤 Токенизация и Морфология
    # ================================================================
    def tokenize(self, text: str) -> List[str]:
        tokens = []
        current_word = []
        for char in text.lower():
            if char.isalnum():
                current_word.append(char)
            else:
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
        if current_word:
            tokens.append(''.join(current_word))
        for token in tokens:
            self.word_occurrence_count[token] = self.word_occurrence_count.get(token, 0) + 1
            w_bytes = token.encode('utf-8')
            if w_bytes not in self.word_forms:
                self.word_forms[w_bytes] = []
            if w_bytes not in self.word_forms[w_bytes]:
                self.word_forms[w_bytes].append(w_bytes)
        return tokens

    def _find_best_lcs(self, word_bytes: bytes) -> bytes:
        best_lcs = b''
        limit = self.calibration.lcs_search_limit
        candidates = list(self.known_words)[:limit]
        for other in candidates:
            if other == word_bytes:
                continue
            lcs = self.trie.find_lcs([word_bytes, other])
            if len(lcs) > len(best_lcs):
                best_lcs = lcs
        return best_lcs

    def _resolve_label(self, word_str: str) -> str:
        if word_str in self._label_cache:
            return self._label_cache[word_str]

        w_bytes = word_str.encode('utf-8')
        if w_bytes not in self.known_words:
            self.known_words.add(w_bytes)
            self.trie.insert(w_bytes)

        # НОВЫЙ ПОДХОД: стемминг вместо LCS
        stem = self.stemmer.stem(word_str)

        if stem != word_str and len(stem) >= self.calibration.min_root_len:
            result = f"root:{stem}"
        else:
            result = word_str.lower()

        self._label_cache[word_str] = result
        return result

    # 🆕 Приоритет 2.2: Детекция намерения вопроса
    def _detect_query_intent(self, text: str, words: List[str]) -> Optional[str]:
        """Определяет тип вопроса."""
        has_question_mark = '?' in text
        for intent_name, marker_set in self.QUESTION_INTENTS.items():
            for w in words:
                if w in marker_set:
                    return intent_name
        if has_question_mark:
            return "what"
        return None

    # ================================================================
    # 💉 Инжекция текста и Опровержение
    # ================================================================
    def _lemmatize(self, word: str) -> str:
        """🆕 Шаг 1.3: Простая лемматизация через существующий stemmer мембраны."""
        return self.stemmer.stem(word.lower())

    # ================================================================
    # 🆕 ФАЗА 2 (Шаг 2.2): Обработка слов-операторов (идея #7)
    # ================================================================
    def _op_node(self, op_type: str):
        """Операторный узел op:<op_type> из реестра Кристалла (или None)."""
        return self.lattice.operator_nodes.get(op_type)

    def _apply_operator(self, op_word: str, context_concepts: list) -> Optional[str]:
        """
        Применить оператор к текущему контексту концептов (идея #7).
        Возвращает тип операции — чтобы inject_text мог синхронизировать
        существующие флаги пайплайна (negation_pending / pending_is_a).
        Часть операций дублирует работу маркерного прохода ниже (contrast/
        condition/cause/is_a/part_of) — connect() идемпотентен, это безопасно.
        """
        op_type = self.OPERATORS.get(op_word.lower())
        if not op_type:
            return None

        tick = self.lattice.tick_count
        reason = lambda src: CrystalReason(
            kind="input", source_label=src, source_type="text",
            context=self.lattice.active_context, timestamp=tick)

        # 🆕 Контекст приходит из valid_concepts — это НЕ всегда метки узлов
        # (например "яблоко" резолвится в "root:яблок"). Разрешаем каждую метку
        # через label_to_id с fallback на root:-форму; неразрешённые отбрасываем.
        ctx_ids: list = []
        for lbl in context_concepts:
            rid = self.lattice.label_to_id.get(lbl) or self.lattice.label_to_id.get(f"root:{lbl}")
            if rid is not None:
                ctx_ids.append(rid)

        if op_type == 'negate':
            # Следующий концепт получит state = FALSE (флаг ставит inject_text)
            self._pending_negate = True

        elif op_type == 'bind':
            # 🆕 Двухпроходная логика (Фаза 2, критерий готовности):
            # в однопроходном цикле 'и' видит только ЛЕВЫЙ концепт (правый ещё
            # не материализован). Запоминаем левый и закрываем пару в
            # _finalize_operator_state после основного цикла.
            self._pending_bind = context_concepts[-1] if context_concepts else ''

        elif op_type == 'branch':
            # 'или' — развилка: отложенная ANALOG-связь пары (A или B)
            self._pending_branch = context_concepts[-1] if context_concepts else ''

        elif op_type == 'contrast':
            # Конфликт как знание (идея #34): EDGE_EXCEPT с весом 80
            if len(ctx_ids) >= 2:
                la = self.lattice.resonators[ctx_ids[-2]].label
                lb = self.lattice.resonators[ctx_ids[-1]].label
                self.lattice.connect(la, lb,
                                     weight=80, edge_type=EDGE_EXCEPT,
                                     reason=reason(f"{la} {op_word} {lb}"))

        elif op_type == 'condition':
            # Следующая пара: A если B → EDGE_COND(B, A)
            self._pending_condition = True

        elif op_type == 'consequence':
            # 'то'/'поэтому': предыдущий концепт — причина следующего
            if len(ctx_ids) >= 2:
                la = self.lattice.resonators[ctx_ids[-2]].label
                lb = self.lattice.resonators[ctx_ids[-1]].label
                self.lattice.connect(la, lb,
                                     weight=self.calibration.causal_marker_weight,
                                     edge_type=EDGE_CAUSE,
                                     reason=reason(f"{la} -> {lb} ({op_word})"))

        elif op_type in ('container', 'surface'):
            # Пространственные отношения: предлог связывает соседние концепты
            # как часть→целое (X в/на Y ⇒ X PART_OF Y на время предложения)
            if len(ctx_ids) >= 2:
                la = self.lattice.resonators[ctx_ids[-1]].label
                lb = self.lattice.resonators[ctx_ids[-2]].label
                self.lattice.connect(la, lb,
                                     weight=self.calibration.syntagm_weight_direct,
                                     edge_type=EDGE_PART_OF,
                                     reason=reason(f"{la} {op_word} {lb}"))

        elif op_type == 'cause':
            # из-за X → Y: CAUSE(X, Y) — маркерный проход построит связь;
            # флаг запоминаем для усиления
            self._pending_cause = True

        elif op_type == 'is_a':
            # уже реализовано через pending_is_a — оставить
            self._pending_is_a = True

        elif op_type == 'purpose':
            # для X → целевая связь EDGE_GOAL (X — цель последнего концепта)
            self._pending_goal = True
            if ctx_ids:
                goal_r = self._op_node('purpose')
                if goal_r is not None:
                    # помечаем: следующий материализованный концепт — цель
                    # (каноническая метка узла, а не raw-метка контекста)
                    self._goal_target_label = self.lattice.resonators[ctx_ids[-1]].label

        elif op_type == 'part_of':
            # часть/из: X часть Y → EDGE_PART_OF(X, Y)
            if len(ctx_ids) >= 2:
                la = self.lattice.resonators[ctx_ids[-2]].label
                lb = self.lattice.resonators[ctx_ids[-1]].label
                self.lattice.connect(la, lb,
                                     weight=self.calibration.causal_marker_weight,
                                     edge_type=EDGE_PART_OF,
                                     reason=reason(f"{la} {op_word} {lb}"))

        return op_type

    def _finalize_operator_state(self, valid_concepts: list, stats: CausalStats):
        """
        🆕 Шаг 2.2 (двухпроходная логика): «доиграть» отложенные операторы
        после основного цикла, когда оба члена пары уже материализованы.
        - 'A и B'   (bind)      → суперпозиция в working_memory + SYN-ребро BIND
        - 'A или B' (branch)    → EDGE_ANALOG(A, B)
        - 'для X'   (purpose)   → EDGE_GOAL(цель ← субъект)
        - операторные узлы получают энергию за каждое применение (шаг 2.3)
        """
        tick = self.lattice.tick_count

        def resolve(lbl):
            return (self.lattice.label_to_id.get(lbl)
                    or self.lattice.label_to_id.get(f"root:{lbl}"))

        # --- bind: 'A и B' — оба концепта теперь известны ---
        # Правый член пары мог НЕ материализоваться в основном цикле (первое
        # вхождение слова, occurrence_count < 2) — тогда достраиваем его здесь.
        pending_bind = getattr(self, '_pending_bind', '')
        if pending_bind and len(valid_concepts) >= 1:
            right_lbl = next((c for c in reversed(valid_concepts)
                              if c != pending_bind), None)
            if right_lbl is None:
                right_lbl = getattr(self, '_bind_right_candidate', None)
                if right_lbl:
                    r_new = self.lattice.get_or_create(right_lbl)
                    r_new.inject_energy(self.calibration.concept_inject_energy, tick)
                    valid_concepts.append(right_lbl)
            if right_lbl:
                left_lbl = pending_bind
                a, b = resolve(left_lbl), resolve(right_lbl)
                if a is not None and b is not None and a != b:
                    # Объединение последних двух концептов: bundled HDC
                    # (временная суперпозиция, идея #56 из A-класса)
                    bundled = self.lattice.encoder.bundle([
                        self.lattice.resonators[a].hdc_vector,
                        self.lattice.resonators[b].hdc_vector
                    ])
                    self.lattice.working_memory.append(bundled)
                    # Операторный узел 'bind' активируется фактом применения
                    op_r = self._op_node('bind')
                    if op_r is not None:
                        op_r.inject_energy(20, tick, source_ids=[a, b])
                    # SYN-слой (идея #8): языковая связь BIND между токенами
                    # последних вхождений этих концептов в тексте.
                    # Ключи _last_token_by_label — raw-метки из предпрохода;
                    # обобщённый поиск: токен, чья лемматизированная
                    # поверхностная форма совпадает с леммой концепта.
                    def tok_for(lbl, rid):
                        t = self._last_token_by_label.get(lbl)
                        if t is not None:
                            return t
                        target = self._lemmatize(str(lbl).replace('root:', ''))
                        for k, v in self._last_token_by_label.items():
                            vr = resolve(k) or resolve(f"root:{k}")
                            if vr == rid:
                                return v
                            r = self.lattice.resonators.get(v)
                            form = getattr(r, 'lexical_form', None) if r else None
                            if form and self._lemmatize(form) == target:
                                return v
                        return None

                    ta = tok_for(left_lbl, a)
                    tb = tok_for(right_lbl, b)
                    if ta is not None and tb is not None and ta != tb:
                        self.lattice.syn_graph[(ta, tb)] = pack_edge(
                            self.calibration.syntagm_weight_adj, EDGE_SYNTAGM)
                    stats.bind_links += 1
        self._pending_bind = ''

        # --- branch: 'A или B' — развилка как ANALOG-связь ---
        pending_branch = getattr(self, '_pending_branch', '')
        if pending_branch and len(valid_concepts) >= 2:
            right_idx = next((i for i in range(len(valid_concepts) - 1, -1, -1)
                              if valid_concepts[i] != pending_branch), None)
            if right_idx is not None:
                la = resolve(pending_branch)
                lb = resolve(valid_concepts[right_idx])
                if la is not None and lb is not None and la != lb:
                    self.lattice.connect(
                        self.lattice.resonators[la].label,
                        self.lattice.resonators[lb].label,
                        weight=self.calibration.syntagm_weight_direct,
                        edge_type=EDGE_ANALOG)
        self._pending_branch = ''

        if getattr(self, '_pending_goal', False):
            goal_lbl = getattr(self, '_goal_target_label', None)
            if goal_lbl and len(valid_concepts) >= 2:
                subj = next((c for c in reversed(valid_concepts[:-1]) if c != goal_lbl), None)
                if subj:
                    self.lattice.connect(subj, goal_lbl,
                                         weight=self.calibration.causal_marker_weight,
                                         edge_type=EDGE_GOAL,
                                         reason=CrystalReason(
                                             kind="input", source_label=f"{subj} для {goal_lbl}",
                                             source_type="text",
                                             context=self.lattice.active_context,
                                             timestamp=self.lattice.tick_count))
            self._pending_goal = False
            self._goal_target_label = None

    def _build_lexical_layers(self, words: List[str], resolved_labels: List[Optional[str]]):
        """
        🆕 ФАЗА 1 (Шаги 1.3/1.4): Разделение «Слово / Смысл / Сущность» (идея #3).

        1. Для каждого словаупотребления создаётся эфемерный TOKEN ("tok:...").
           Токены уникальны для каждого вхождения — порядок слов в предложении
           различается именно на этом слое.
        2. Лемматизация: tok:банке → lem:банка (LEMMA-узел переиспользуется).
        3. SYN-граф (lattice.syn_graph): связи МЕЖДУ ТОКЕНАМИ — чистая языковая
           структура (идея #8), живёт отдельно от семантического графа.
        4. Семантический граф (connections LEMMA-узлов): связи между леммами —
           НЕ между токенами. "Пётр ударил Ивана" и "Иван ударил Петра" дают
           разные syn_graph-связи, но одинаковые смысловые (лемма-лемма).
        """
        token_ids: List[int] = []
        token_words: List[str] = []
        tick = self.lattice.tick_count

        # 1. TOKEN-слой: каждый токен ЭФЕМЕРЕН — уникален для каждого вхождения
        # (один и тот же "иван" в разных предложениях = разные узлы, иначе
        # syn_graph не различал бы порядок слов).
        seq = getattr(self, '_token_seq', 0) + 1
        self._token_seq = seq + len(words)  # резервируем диапазон под это предложение
        for w, lbl in zip(words, resolved_labels):
            if lbl is None:
                continue  # стоп-слова/операторы не материализуются даже как токены
            tok_label = f"tok:{w.lower()}@{tick}#{seq}"
            seq += 1
            token_r = self.lattice.get_or_create(tok_label, NodeType.TOKEN)
            token_r.lexical_form = w.lower()  # 🆕 Шаг 1.1: поверхностная форма слова
            token_r.last_tick = tick          # эфемерные: живут до cleanup_tokens()
            token_ids.append(token_r.id)
            token_words.append(w.lower())

        # 2. Лемматизация: tok:банке → lem:банка; TOKEN → LEMMA (lemma_id)
        for tok_id, tok, (w, lbl) in zip(token_ids, token_words,
                                         [(w, l) for w, l in zip(words, resolved_labels) if l is not None]):
            lemma = self._lemmatize(tok)
            if not lemma:
                continue
            lemma_r = self.lattice.get_or_create(f"lem:{lemma}", NodeType.LEMMA)
            token_r = self.lattice.resonators[tok_id]
            token_r.lemma_id = lemma_r.id
            # 🆕 Фаза 2 (bind): запоминаем последний токен каждого концепта —
            # оператор 'и' строит по ним SYN-ребро BIND в syn_graph
            if lbl:
                self._last_token_by_label[lbl] = tok_id
            # Активация леммы следует за активацией её токена (смысл ≠ слово)
            src_r = self.lattice.resonators[tok_id]
            if src_r.is_active():
                lemma_r.inject_energy(src_r.energy // 2, tick, source_ids=[src_r.id])

        # 3. SYN-граф: связи между ТОКЕНАМИ в предложении (только язык!)
        for i in range(len(token_ids) - 1):
            packed = pack_edge(self.calibration.syntagm_weight_adj, EDGE_SYNTAGM)
            self.lattice.syn_graph[(token_ids[i], token_ids[i + 1])] = packed

        # 4. Семантический граф: связи между леммами (НЕ токенами!)
        lemma_ids = [self.lattice.resonators[tid].lemma_id for tid in token_ids]
        lemma_ids = [lid for lid in lemma_ids if lid is not None]
        for i in range(len(lemma_ids) - 1):
            r1 = self.lattice.resonators[lemma_ids[i]]
            r2 = self.lattice.resonators[lemma_ids[i + 1]]
            packed = pack_edge(self.calibration.syntagm_weight_adj, EDGE_SYNTAGM)
            r1.connections[r2.id] = packed
            r2.connections[r1.id] = packed  # SYN — двунаправленная связь

    def inject_text(self, text: str) -> CausalStats:
        words = self.tokenize(text)
        # 🆕 Фаза 2: кандидаты парных операторов — только в рамках предложения
        self._bind_right_candidate = None
        stats = CausalStats()
        resolved_labels = []
        valid_concepts = []

        # 🆕 Приоритет 2.2: Детекция намерения
        intent = self._detect_query_intent(text, words)
        if intent:
            self.last_query_intent = intent
            stats.questions += 1
            # Создаём узел намерения
            intent_label = f"intent:{intent}"
            r_intent = self.lattice.get_or_create(intent_label)
            r_intent.inject_energy(self.calibration.concept_inject_energy, self.lattice.tick_count)
            self.lattice.set_belief_status(
                r_intent.id, "observed", 1.0,
                CrystalReason(kind="input", source_label=intent_label, source_type="text",
                              context=self.lattice.active_context, timestamp=self.lattice.tick_count)
            )

        # 🆕 Приоритет 2.1: Флаг отрицания
        negation_pending = False
        pending_is_a = False  # 🆕 'это' — оператор связки X IS_A Y

        # 🆕 ФАЗА 2 (Шаг 1.3/2.2): SYN-граф строится ДО применения операторов:
        # bind-оператору нужны id токенов, а _last_token_by_label заполняется
        # внутри _build_lexical_layers. Связи в syn_graph идемпотентны (key-value),
        # повторное применение оператора безопасно.
        pre_labels: List[Optional[str]] = []
        for w in words:
            if w.lower() in self.OPERATORS or (
                    w in self.STOP_WORDS and
                    w not in self.CAUSE_MARKERS and
                    w not in self.EXCEPT_MARKERS and
                    w not in self.COND_MARKERS and
                    w not in self.IS_A_MARKERS):
                pre_labels.append(None)
            else:
                pre_labels.append(self._resolve_label(w))
        self._build_lexical_layers(words, pre_labels)

        for w in words:
            # 🆕 ФАЗА 2, Шаг 2.2 (идея #7): служебные слова — ОПЕРАТОРЫ, не концепты.
            # Перехватываем ДО стоп-листа; контекст (valid_concepts) к этому
            # моменту уже наполнен предыдущими словами цикла.
            if w.lower() in self.OPERATORS:
                resolved_labels.append(None)     # оператор НЕ материализуется как узел
                op_type = self._apply_operator(w.lower(), valid_concepts)
                # Совместимость с существующими флагами пайплайна:
                if op_type == 'negate':
                    negation_pending = True      # следующий концепт → FALSE/defeated
                    stats.negations += 1
                elif op_type == 'is_a':
                    pending_is_a = True          # X это Y → EDGE_IS_A
                continue

            # Пропускаем обычные стоп-слова (кроме маркеров связок)
            if (w in self.STOP_WORDS and
                w not in self.CAUSE_MARKERS and
                w not in self.EXCEPT_MARKERS and
                w not in self.COND_MARKERS and
                w not in self.IS_A_MARKERS):
                resolved_labels.append(None)
                continue

            # 🆕 Фикс: после 'это' или 'не' принудительно материализуем следующий концепт
            force_materialize = pending_is_a or negation_pending

            label = self._resolve_label(w)
            resolved_labels.append(label)
            occurrence_count = self.word_occurrence_count.get(w, 0)

            # 🆕 Фаза 2: висит отложенный bind/branch ('и'/'или' уже пройдено) —
            # следующий концепт принудительно материализуется как правый член пары.
            if (getattr(self, '_pending_bind', '') or
                    getattr(self, '_pending_branch', '')):
                force_materialize = True

            should_materialize = (
                force_materialize or          # ← ДОБАВЛЕНО
                occurrence_count >= 2 or
                label in self.lattice.label_to_id or
                label.startswith("root:")
            )
            if should_materialize:
                r = self.lattice.get_or_create(label)
                if r.state == TruthValue.FALSE:
                    r.state = TruthValue.PARADOX
                elif r.state == TruthValue.VOID:
                    r.state = TruthValue.TRUE
                r.inject_energy(self.calibration.concept_inject_energy, self.lattice.tick_count)
                valid_concepts.append(label)

                # 🆕 Фаза 2: если впереди висит отложенный bind/branch ('и'/'или'
                # ужеSeen, левый член запомнен), этот концепт — правый член пары.
                # Запоминаем кандидата (материализация ниже произойдёт в любом
                # случае через get_or_create в _finalize при необходимости).
                if getattr(self, '_pending_bind', '') or getattr(self, '_pending_branch', ''):
                    self._bind_right_candidate = label

                # 🆕 Приоритет 2.1: Если было "не" — инвертируем концепт
                if negation_pending:
                    self.lattice.set_belief_status(
                        r.id, "defeated", 1.0,
                        CrystalReason(kind="input", source_label="NEGATION", source_type="text",
                                      context=self.lattice.active_context, timestamp=self.lattice.tick_count)
                    )
                    # Связь EXCEPT от предыдущего концепта к инвертированному
                    if len(valid_concepts) >= 2:
                        prev_label = valid_concepts[-2]
                        self.lattice.connect(
                            prev_label, label,
                            weight=self.calibration.except_antonym_weight,
                            edge_type=EDGE_EXCEPT,
                            reason=CrystalReason(
                                kind="input", source_label=f"не {label}", source_type="text",
                                context=self.lattice.active_context, timestamp=self.lattice.tick_count)
                        )
                        stats.except_links += 1
                    negation_pending = False

        for i in range(len(valid_concepts) - 1):
            self.lattice.connect(
                valid_concepts[i], valid_concepts[i+1],
                weight=self.calibration.syntagm_weight_direct,
                edge_type=EDGE_SYNTAGM
            )
            stats.syntagm_links += 1
            if i + 2 < len(valid_concepts):
                self.lattice.connect(
                    valid_concepts[i], valid_concepts[i+2],
                    weight=self.calibration.syntagm_weight_skip,
                    edge_type=EDGE_SYNTAGM
                )

        # 🆕 Обработка паттерна "X это Y" → IS_A (оператор связки, без узла 'это')
        if pending_is_a and len(valid_concepts) >= 2:
            subj = valid_concepts[-2]  # X
            obj = valid_concepts[-1]   # Y
            reason = CrystalReason(
                kind="input",
                source_label=f"{subj} IS_A {obj}",
                source_type="text",
                confidence=0.95,
                context=self.lattice.active_context,
                timestamp=self.lattice.tick_count,
                metadata={"relation": "is_a", "marker": "это"}
            )
            self.lattice.connect(subj, obj, weight=self.calibration.causal_marker_weight,
                                 edge_type=EDGE_IS_A, reason=reason)
            stats.is_a_links += 1

        pending_is_a = False

        # 🆕 ФАЗА 2 (Шаг 2.2): доиграть отложенные операторы ('для' → EDGE_GOAL)
        # (SYN-граф построен раньше — перед циклом операторов, шаг 1.3/2.2)
        self._finalize_operator_state(valid_concepts, stats)

        self._learn_contextual_rules(words, resolved_labels)

        # 🆕 Приоритет 2.3: Извлечение примитивных триплетов с маркерами
        for i in range(len(words)):
            marker = words[i]
            edge_type = None
            relation_meta = None

            if marker in self.CAUSE_MARKERS:
                edge_type = EDGE_CAUSE
            elif marker in self.EXCEPT_MARKERS:
                edge_type = EDGE_EXCEPT
            elif marker in self.COND_MARKERS:
                edge_type = EDGE_COND
            elif marker in self.IS_A_MARKERS:
                edge_type = EDGE_SYNTAGM
                relation_meta = "is_a"

            # 🆕 Задача 2: оператор связки 'это' (не входит в IS_A_MARKERS как концепт)
            if marker == 'это':
                edge_type = EDGE_IS_A
                relation_meta = "is_a"

            if edge_type is not None:
                left_concept = None
                for j in range(i - 1, max(-1, i - 4), -1):
                    if j < len(resolved_labels) and resolved_labels[j]:
                        left_concept = resolved_labels[j]
                        break
                right_concept = None
                for j in range(i + 1, min(len(words), i + 4)):
                    if j < len(resolved_labels) and resolved_labels[j]:
                        right_concept = resolved_labels[j]
                        break
                if left_concept and right_concept:
                    reason = CrystalReason(
                        kind="input",
                        source_label=f"{left_concept}->{right_concept}",
                        source_type="text",
                        confidence=0.9,
                        context=self.lattice.active_context,
                        timestamp=self.lattice.tick_count,
                        metadata={"marker": marker, **({"relation": relation_meta} if relation_meta else {})}
                    )
                    self.lattice.connect(
                        left_concept, right_concept,
                        weight=self.calibration.causal_marker_weight,
                        edge_type=edge_type,
                        reason=reason
                    )
                    if edge_type == EDGE_CAUSE: stats.cause_links += 1
                    elif edge_type == EDGE_EXCEPT: stats.except_links += 1
                    elif edge_type == EDGE_COND: stats.cond_links += 1
                    elif relation_meta == "is_a": stats.is_a_links += 1

                    # 🆕 Приоритет 2.1: Дополнительная обработка EXCEPT для антонимов
                    if edge_type == EDGE_EXCEPT:
                        if i > 0 and i + 1 < len(words):
                            left_word = words[i-1] if i > 0 else None
                            right_word = words[i+1] if i + 1 < len(words) else None
                            if left_word and right_word:
                                left_lbl = self._resolve_label(left_word)
                                right_lbl = self._resolve_label(right_word)
                                if left_lbl and right_lbl and left_lbl != right_lbl:
                                    self.lattice.connect(
                                        left_lbl, right_lbl,
                                        weight=self.calibration.except_antonym_weight,
                                        edge_type=EDGE_EXCEPT
                                    )

        print(f"🧬 [МЕМБРАНА] Словарь: {len(self.known_words)} словоформ.")
        return stats

    def inject_defeater(self, text: str):
        words = self.tokenize(text)
        for i, w in enumerate(words):
            label = w.lower()
            w_bytes = w.encode('utf-8')
            if w_bytes in self.known_words and len(self.known_words) > 1:
                lcs = self._find_best_lcs(w_bytes)
                lcs_str = lcs.decode('utf-8', errors='ignore')
                if len(lcs_str) >= self.calibration.min_root_len and lcs in w_bytes:
                    root_idx = w_bytes.find(lcs)
                    if root_idx == 0:
                        label = f"root:{lcs_str}"
            r = self.lattice.get_or_create(label)
            if r.state == TruthValue.TRUE:
                r.state = TruthValue.PARADOX
            elif r.state == TruthValue.VOID:
                r.state = TruthValue.FALSE
            r.energy = 0
            r.last_tick = self.lattice.tick_count

    # ================================================================
    # 🗣️ Генерация речи
    # ================================================================
    def generate_text(self, seed_concepts: List[str], max_length: int = 15, beam_width: int = 2) -> str:
        seed_ids = []
        intention_vectors = []
        seed_concepts_clean = []
        for c in seed_concepts:
            if not c: continue
            resolved = self.resolve_query(c)
            # 🆕 Фикс: пробуем найти и с префиксом root:
            if resolved not in self.lattice.label_to_id:
                if f"root:{resolved}" in self.lattice.label_to_id:
                    resolved = f"root:{resolved}"
                elif c in self.lattice.label_to_id:
                    resolved = c
                elif f"root:{c}" in self.lattice.label_to_id:
                    resolved = f"root:{c}"
            if resolved in self.lattice.label_to_id:
                seed_ids.append(self.lattice.label_to_id[resolved])
                intention_vectors.append(self.lattice.resonators[self.lattice.label_to_id[resolved]].hdc_vector)
                clean = self._get_surface_form(resolved) if resolved.startswith("root:") else resolved
                if clean not in self.STOP_WORDS and len(clean) >= self.calibration.speech_min_word_len:
                    seed_concepts_clean.append(clean)
            elif c in self.lattice.label_to_id:
                seed_ids.append(self.lattice.label_to_id[c])
                intention_vectors.append(self.lattice.resonators[self.lattice.label_to_id[c]].hdc_vector)
                clean = c[5:] if c.startswith("root:") else c
                if clean not in self.STOP_WORDS and len(clean) >= self.calibration.speech_min_word_len:
                    seed_concepts_clean.append(clean)
        if not intention_vectors or not seed_ids:
            return "Трасса обрывается. Энтропия высока."
        intention_vector = self.lattice.encoder.bundle(intention_vectors)
        self.lattice.blackboard.intention_vector = intention_vector

        hypotheses = []
        plan1_ids = self.lattice.build_conceptual_plan(seed_ids, horizon=max_length)
        if plan1_ids:
            score1 = self.lattice.score_hypothesis(plan1_ids, seed_ids)
            hypotheses.append((plan1_ids, score1, "standard"))
        path2_ids = self.lattice.tick_backward(seed_ids[0], max_depth=max_length, prefer_syn=False)
        plan2_ids = [pid for pid in path2_ids if pid in self.lattice.resonators]
        if plan2_ids and plan2_ids != plan1_ids:
            score2 = self.lattice.score_hypothesis(plan2_ids, seed_ids)
            hypotheses.append((plan2_ids, score2, "causal"))
        path3_ids = self.lattice.tick_backward(seed_ids[0], max_depth=max(3, max_length // 2), prefer_syn=True)
        plan3_ids = [pid for pid in path3_ids if pid in self.lattice.resonators]
        if plan3_ids and plan3_ids != plan1_ids and plan3_ids != plan2_ids:
            score3 = self.lattice.score_hypothesis(plan3_ids, seed_ids)
            hypotheses.append((plan3_ids, score3, "short"))
        if not hypotheses:
            return "Тишина. Не удалось сформировать ни одной гипотезы."
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        best_plan_ids, best_score, strategy = hypotheses[0]

        self.lattice.evoke_frames()
        dep_tree = self.lattice.build_dependency_tree(best_plan_ids)
        raw_words = []
        if dep_tree:
            raw_words = dep_tree.linearize(self.lattice)
        if not raw_words:
            raw_words = [self.lattice.resonators[pid].label[5:] if self.lattice.resonators[pid].label.startswith("root:") else self.lattice.resonators[pid].label
                         for pid in best_plan_ids if pid in self.lattice.resonators]
            raw_words = [w for w in raw_words if w and not w.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')) and w != 'SELF']
        if not raw_words:
            return "Мысль не обрела форму."

        primary_seed_id = seed_ids[0]
        primary_r = self.lattice.resonators.get(primary_seed_id)
        sentence = ""
        if primary_r:
            primary_lbl = self._get_surface_form(primary_r.label)
            # 🆕 ЗАДАЧА 7: выбор конструкции по типу сильнейшего отношения
            best_relation = None
            best_target = None
            best_weight = 0

            best_target_id = None
            for tgt_id, packed in primary_r.connections.items():
                w, et = unpack_edge(packed)
                if tgt_id not in self.lattice.resonators:
                    continue
                tgt_r = self.lattice.resonators[tgt_id]
                if not tgt_r.is_active():
                    continue
                # 🆕 Фикс: служебные абстракции не подходят как объект высказывания
                if tgt_r.label.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')) or '->' in tgt_r.label:
                    continue
                tgt_lbl = self._get_surface_form(tgt_r.label)
                if len(tgt_lbl) < 3 or tgt_lbl == primary_lbl:
                    continue
                if w > best_weight:
                    best_weight = w
                    best_target = tgt_lbl
                    best_target_id = tgt_id
                    best_relation = et

            if best_relation == EDGE_IS_A:
                entity_display = self._get_surface_form(primary_r.label)
                class_display = (self._get_surface_form(self.lattice.resonators[best_target_id].label)
                                 if best_target_id in self.lattice.resonators else best_target)
                sentence = self.CONSTRUCTIONS["IS_A"].format(
                    Entity=entity_display.capitalize(), Class=class_display)
            elif best_relation == EDGE_CAUSE:
                sentence = self.CONSTRUCTIONS["CAUSATION"].format(
                    Cause=primary_lbl.capitalize(), Effect=best_target)
            elif best_relation == EDGE_PART_OF:
                sentence = self.CONSTRUCTIONS["PART_OF"].format(
                    Part=primary_lbl.capitalize(), Whole=best_target)
            elif best_relation == EDGE_EXCEPT:
                sentence = self.CONSTRUCTIONS["NEGATION"].format(
                    Entity=primary_lbl.capitalize(), Class=best_target)
            elif best_target:
                ADJ_ENDINGS = ('ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие')
                if best_target.endswith(ADJ_ENDINGS):
                    sentence = self.CONSTRUCTIONS["STATE_ADJ"].format(
                        Entity=primary_lbl.capitalize(), Property=best_target)
                else:
                    sentence = self.CONSTRUCTIONS["STATE_NOUN"].format(
                        Entity=primary_lbl.capitalize(), Property=best_target)
        if not sentence:
            analogies = self.lattice.find_analogies(primary_lbl)
            if analogies:
                top_analog = analogies[0][0]
                sentence = f"{primary_lbl.capitalize()} похоже на {self._get_surface_form(top_analog)}."
            elif len(raw_words) >= 2:
                subj = raw_words[0]
                VERB_ENDINGS = ('ть', 'ти', 'чь', 'ют', 'ут', 'ат', 'ят', 'ит', 'ет', 'ла', 'ли', 'ло', 'лся', 'лась', 'лись', 'лось', 'утся', 'ятся')
                obj = next((w for w in raw_words[1:] if not w.endswith(VERB_ENDINGS) and w != subj), None)
                if obj:
                    sentence = f"{subj.capitalize()} это {obj}."
                else:
                    sentence = f"{subj.capitalize()} {raw_words[1]}."
            else:
                sentence = " ".join(raw_words).capitalize() + "."
        sentence = self._apply_self_critic_text(sentence, best_plan_ids)
        return sentence.replace("  ", " ").strip()

    def _apply_self_critic_text(self, text: str, plan_ids: List[int]) -> str:
        words = [w.strip(".,!?-—").lower() for w in text.split() if w.strip(".,!?-—")]
        if len(words) < 2:
            return text
        mapped_nodes = []
        for w in words:
            if w in self.STOP_WORDS:
                continue
            r = self.lattice.find_best_resonator_by_label(w)
            if r:
                mapped_nodes.append((w, r))
        if len(mapped_nodes) < 2:
            return text
        connected_pairs = 0
        total_pairs = 0
        plan_set = set(plan_ids)
        for i in range(len(mapped_nodes) - 1):
            w1, r1 = mapped_nodes[i]
            w2, r2 = mapped_nodes[i+1]
            total_pairs += 1
            if r2.id in r1.connections or r1.id in r2.connections:
                connected_pairs += 1
            else:
                n1 = set(r1.connections.keys())
                n2 = set(r2.connections.keys())
                union = len(n1 | n2)
                intersection = len(n1 & n2)
                if union > 0 and (intersection / union) > 0.05:
                    connected_pairs += 1
                else:
                    if r1.id in plan_set and r2.id in plan_set:
                        connected_pairs += 1
        # 🆕 ЗАДАЧА 7: снижен порог связности 0.40 -> 0.25 (меньше ложных "Мысль фрагментарна")
        if total_pairs > 0 and (connected_pairs / total_pairs) < 0.25:
            return "Мысль фрагментарна. Требуется больше фактов."
        return text