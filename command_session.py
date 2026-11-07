"""Shared terminal/web dispatcher extracted from the original REPL.
Only transport and state ownership moved; the command branches are shared.
"""
from main import (CrystalLattice, CrystalPopulation, pack_edge, unpack_edge, EDGE_NAMES,
                  EDGE_CAUSE, EDGE_EXCEPT, EDGE_ANALOG, EDGE_GOAL, CrystalReason,
                  LanguageMembrane, CrystalSnapshot, CalibrationProfile,
                  print_status, wiki_probe, autonomous_research)
import traceback
import os
import json
import contextlib
import io
from dataclasses import dataclass

@dataclass
class CommandResult:
    output: str
    keep_running: bool = True
    error: bool = False

class CommandSession:
    def __init__(self, population=None, membrane=None):
        self.population = population if population is not None else CrystalPopulation()
        self.membrane = membrane if membrane is not None else LanguageMembrane(self.population.active)
        self.last_state_cache = None
        self.last_compare_all_antagonisms = []
        self.last_compare_antagonisms = []
        self.last_compare_entities = None

    def execute(self, command, capture=True):
        if not command.strip():
            return CommandResult('')
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            if capture:
                stack.enter_context(contextlib.redirect_stdout(output))
                stack.enter_context(contextlib.redirect_stderr(output))
            try:
                keep = self._run(command.strip())
            except Exception:
                traceback.print_exc()
                return CommandResult(output.getvalue(), error=True)
        return CommandResult(output.getvalue(), keep is not False)

    def _run(self, cmd):
        population = self.population
        membrane = self.membrane
        last_state_cache = self.last_state_cache
        last_compare_all_antagonisms = self.last_compare_all_antagonisms
        last_compare_antagonisms = self.last_compare_antagonisms
        last_compare_entities = self.last_compare_entities
        try:
            try:
                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()
                if action.startswith('!'):
                    complaint = action[1:]
                    if population.active.calibration.feedback(complaint):
                        return
                    else:
                        print(f"❓ Неизвестная оценка '!{complaint}'.")
                        print(f'   Доступные: !бессвязно !мимо !кратко !длинно !сухо !бред !ok')
                        return
                args = parts[1] if len(parts) > 1 else ''
                if action == 'выход':
                    print('💎 Кристалл засыпает...')
                    return False
                elif action == 'ввод':
                    if not args:
                        print('⚠️  Использование: ввод <текст>')
                        return
                    stats = membrane.inject_text(args)
                    print(f"👁 [ВВОД] Инжектировано в '{args}'")
                    if population.check_population_dynamics():
                        membrane.lattice = population.active
                elif action == 'ложь':
                    if not args:
                        print('⚠️  Использование: ложь <текст>')
                        return
                    membrane.inject_defeater(args)
                    print(f"🛡 [ОПРОВЕРЖЕНИЕ] Инжектировано в '{args}'")
                elif action == 'отрицание':
                    parts_arg = args.split()
                    if len(parts_arg) != 2:
                        print('⚠️  Использование: отрицание <А> <Б>')
                        return
                    a, b = parts_arg
                    population.active.add_defeater(a, b)
                    print(f"🔗 Создана связь отрицания: '{a}' -> NOT '{b}'")
                elif action == 'думай':
                    n = int(args) if args.isdigit() else 1
                    for _ in range(n):
                        population.active.tick()
                    print(f'⚡ [МЫШЛЕНИЕ] Прошло {n} тактов (текущий такт: {population.active.tick_count})')
                    print_status(population.active)
                    if population.check_population_dynamics():
                        membrane.lattice = population.active
                elif action == 'отжиг':
                    print('🔥 [ОТЖИГ] Запуск термического отжига...')
                    count = population.active.anneal_paradoxes()
                    if count > 0:
                        print(f'   ⚡ Мутировано узлов: {count}.')
                    else:
                        print('   ⚪ Парадоксов не обнаружено.')
                    print_status(population.active)
                elif action == 'состояние':
                    print_status(population.active)
                elif action == 'сохрани':
                    fname = args if args else 'snapshot'
                    if not fname.endswith('.cry'):
                        fname += '.cry'
                    CrystalSnapshot.save(population.active, fname)
                    last_state_cache = CrystalSnapshot.generate_state_cache(population.active)
                elif action == 'мысль':
                    if not args:
                        print('⚠️ Использование: мысль <концепт>')
                        return
                    print(f"🧠 [МЫСЛИТЕЛЬНЫЙ ПРОЦЕСС] Формирование концептуального плана для '{args}'...")
                    query_label = membrane.resolve_query(args.strip())
                    seed_ids = []
                    if query_label in population.active.label_to_id:
                        seed_ids.append(population.active.label_to_id[query_label])
                    elif f'root:{query_label}' in population.active.label_to_id:
                        seed_ids.append(population.active.label_to_id[f'root:{query_label}'])
                    if not seed_ids:
                        print('   ⚪ Концепт не найден в памяти.')
                        return
                    plan_ids = population.active.build_conceptual_plan(seed_ids, horizon=10)
                    population.active.evoke_frames()
                    print(f'\n📋 КОНЦЕПТУАЛЬНЫЙ ПЛАН (ЧТО сказать):')
                    plan_labels = []
                    for pid in plan_ids:
                        if pid in population.active.resonators:
                            lbl = population.active.resonators[pid].label
                            clean = lbl[5:] if lbl.startswith('root:') else lbl
                            if not lbl.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')) and lbl != 'SELF':
                                plan_labels.append(clean)
                    print(f"   🔗 {(' -> '.join(plan_labels) if plan_labels else 'Пусто')}")
                    print(f'\n🎭 АКТИВНЫЕ ФРЕЙМЫ:')
                    active_frames_found = False
                    for fid, frame in population.active.active_frames.items():
                        if frame.activation > 0.1:
                            active_frames_found = True
                            filled = frame.get_filled_roles()
                            filled_str = []
                            for role, node_id in filled.items():
                                if node_id in population.active.resonators:
                                    lbl = population.active.resonators[node_id].label
                                    clean = lbl[5:] if lbl.startswith('root:') else lbl
                                    filled_str.append(f'{role}={clean}')
                            print(f"   🎬 [{frame.label:<12}] Активация: {frame.activation:.2f} | Роли: {', '.join(filled_str) or 'Пусты'}")
                    if not active_frames_found:
                        print('   ⚪ Фреймы не эвоцированы.')
                elif action == 'фреймы':
                    print('🎭 [ФРЕЙМЫ] Состояние символических структур:')
                    active_frames_found = False
                    for fid, frame in population.active.active_frames.items():
                        filled = frame.get_filled_roles()
                        if filled or frame.activation > 0:
                            active_frames_found = True
                            filled_str = []
                            for role, node_id in filled.items():
                                if node_id in population.active.resonators:
                                    lbl = population.active.resonators[node_id].label
                                    clean = lbl[5:] if lbl.startswith('root:') else lbl
                                    filled_str.append(f'{role}={clean}')
                                else:
                                    filled_str.append(f'{role}=?')
                            print(f"   🎬 [{frame.label:<12}] Активация: {frame.activation:.2f} | {', '.join(filled_str) or 'Слоты пусты'}")
                    if not active_frames_found:
                        print('   ⚪ Все фреймы неактивны.')
                elif action == 'объясни':
                    path = getattr(population.active, 'last_query_path', None)
                    if not path:
                        print("⚠️ Нет последнего пути запроса. Сначала используйте 'спроси'.")
                        return
                    print(population.active.explain_path(path))
                elif action == 'гипотезы':
                    hyps = getattr(population.active, 'hypotheses', {})
                    if not hyps:
                        print('⚪ Активных гипотез нет.')
                        return
                    print(f'🧪 [ГИПОТЕЗЫ] {len(hyps)} активных:')
                    for nid, reason in hyps.items():
                        if nid in population.active.resonators:
                            lbl = population.active.resonators[nid].label
                            print(f'   • {lbl} | уверенность: {reason.confidence:.2f} | источник: {reason.source_type} ({reason.source_label})')
                elif action in ('подтверди', 'опровергни'):
                    if not args:
                        print(f'⚠️ Использование: {action} <узел>')
                        return
                    found_r = population.active.find_best_resonator_by_label(args)
                    if not found_r:
                        print(f"⚪ Узел '{args}' не найден.")
                        return
                    if action == 'подтверди':
                        if found_r.id in population.active.hypotheses:
                            population.active.confirm_hypothesis(found_r.id)
                            print(f"✅ Гипотеза '{found_r.label}' подтверждена (belief_status='confirmed').")
                        else:
                            print(f"⚠️ Узел '{found_r.label}' не является активной гипотезой.")
                    elif found_r.id in population.active.hypotheses:
                        population.active.reject_hypothesis(found_r.id)
                        print(f"❌ Гипотеза '{found_r.label}' отклонена (belief_status='defeated').")
                    else:
                        print(f"⚠️ Узел '{found_r.label}' не является активной гипотезой.")
                elif action == 'контексты':
                    ctxs = getattr(population.active, 'contexts', {})
                    active_ctx = getattr(population.active, 'active_context', 'global')
                    print(f"🗂️  [КОНТЕКСТЫ] Активный: '{active_ctx}' | Всего: {len(ctxs)}")
                    by_parent = {}
                    for name, node in ctxs.items():
                        p = node.parent or '<root>'
                        by_parent.setdefault(p, []).append((name, node))

                    def _print_tree(parent, indent=0):
                        for name, node in by_parent.get(parent, []):
                            marker = ' ◀' if name == active_ctx else ''
                            print(f"   {'  ' * indent}└─ {name} [{node.context_type}] (такт {node.created_tick}){marker}")
                            _print_tree(name, indent + 1)
                    _print_tree('<root>')
                elif action == 'качество':
                    if not args:
                        print('⚠️ Использование: качество <цель>')
                        return
                    found_r = population.active.find_best_resonator_by_label(args)
                    if not found_r:
                        found_r = population.active.get_or_create(args)
                    plan_ids = population.active.build_conceptual_plan([found_r.id], horizon=10)
                    if not plan_ids:
                        print('⚪ План не удалось построить.')
                        return
                    score_info = population.active.score_plan_quality(plan_ids, [found_r.id])
                    plan_labels = [population.active.resonators[pid].label for pid in plan_ids if pid in population.active.resonators]
                    print(f"📊 [ОЦЕНКА ПЛАНА] для '{args}':")
                    print(f"   План: {' -> '.join(plan_labels)}")
                    print(f"   Итоговый score: {score_info['score']:.3f}")
                    print(f"   • Покрытие: {score_info['coverage']:.2f}")
                    print(f"   • Связность: {score_info['connectivity']:.2f}")
                    print(f"   • Причинность: {score_info['causality']:.2f}")
                    print(f"   • Штраф Оккама: {score_info['complexity_penalty']:.2f}")
                    if score_info['warnings']:
                        print(f"   ⚠️ Предупреждения: {', '.join(score_info['warnings'])}")
                elif action == 'изучи' or action == 'автономия':
                    if not args:
                        print('⚠️ Использование: изучи <вопрос/термин>')
                        return
                    query = args.strip()
                    print(f"🔮 [АВТОНОМИЯ] Запуск цикла познания для '{query}'...")
                    autonomous_research(membrane, population.active, query, ticks=20)
                    print_status(population.active)
                    if population.check_population_dynamics():
                        membrane.lattice = population.active
                elif action == 'эпизод':
                    if last_state_cache is None:
                        print("⚠️  Нет базы! Сначала используйте 'сохрани'.")
                        return
                    fname = args if args else 'episode'
                    if not fname.endswith('.cdt'):
                        fname += '.cdt'
                    CrystalSnapshot.save_delta(population.active, fname, last_state_cache)
                    last_state_cache = CrystalSnapshot.generate_state_cache(population.active)
                elif action in {'убери', 'ложный'}:
                    if not args.strip().isdigit():
                        print('⚠️ Использование: убери <номер антагонизма из последнего сравнения>')
                        return
                    n = int(args.strip())
                    if not last_compare_antagonisms or n < 1 or n > len(last_compare_antagonisms):
                        print(f"⚠️ Нет антагонизма #{n}. Сначала выполните 'сравнение'.")
                        return
                    a1, a2 = last_compare_antagonisms[n - 1]
                    population.active.mark_false_antagonism(a1, a2)

                    def _d(lbl):
                        return lbl.split(':', 1)[1] if lbl.startswith('root:') else lbl
                    print(f"   ✅ Антагонизм '{_d(a1)}' ↔ '{_d(a2)}' помечен как ложный.")
                    remaining = [pair for pair in last_compare_all_antagonisms if pair not in last_compare_antagonisms and (not population.active._is_false_antagonism(pair[0], pair[1]))]
                    if not remaining:
                        print('   ⚪ Больше нет подходящих скрытых кандидатов.')
                        last_compare_antagonisms.pop(n - 1)
                    else:
                        print(f'\n   🔄 Выберите замену из скрытых кандидатов:')
                        print(f'      0. Ничего не подходит')
                        max_choice = min(5, len(remaining))
                        for idx, (alt1, alt2) in enumerate(remaining[:max_choice], 1):
                            print(f'      {idx}. 🔴 {last_compare_entities[0]}: {_d(alt1)}  <-->  {last_compare_entities[1]}: {_d(alt2)}')
                        try:
                            choice = input(f'   👉 Ваш выбор (0-{max_choice}): ').strip()
                            if choice.isdigit():
                                c = int(choice)
                                if 1 <= c <= max_choice:
                                    chosen = remaining[c - 1]
                                    last_compare_antagonisms[n - 1] = chosen
                                    population.active.connect(chosen[0], chosen[1], weight=population.active.calibration.except_antonym_weight, edge_type=EDGE_EXCEPT)
                                    print(f'   🧬 Замена произведена!')
                                elif c == 0:
                                    last_compare_antagonisms.pop(n - 1)
                                    print('   🗑️ Антагонизм просто удален.')
                                else:
                                    last_compare_antagonisms.pop(n - 1)
                        except Exception as e:
                            print(f'   ⚠️ Ошибка ввода: {e}')
                            last_compare_antagonisms.pop(n - 1)
                elif action == 'загрузи':
                    fname = args if args else 'snapshot'
                    if not fname.endswith('.cry') and (not fname.endswith('.cdt')):
                        fname += '.cry'
                    if CrystalSnapshot.load(population.active, fname, last_state_cache):
                        last_state_cache = CrystalSnapshot.generate_state_cache(population.active)
                        print_status(population.active)
                elif action == 'сон':
                    print('💤 Кристалл погружается в сон...')
                    merged, pruned, dreams = population.active.defragment()
                    print(f'🧹 [СОН] Завершено. Слияний: {merged}, Удалено: {pruned}')
                    if dreams:
                        for d in dreams:
                            print(f'   {d}')
                    print_status(population.active)
                elif action in {'сравни', 'сравнение', 'отличи'}:
                    parts_arg = args.split()
                    if len(parts_arg) < 3 or parts_arg[1] not in ['от', 'и', 'с']:
                        print('⚠️ Использование: сравни <А> от <Б>')
                        return
                    label1 = parts_arg[0]
                    label2 = parts_arg[2]
                    print(f"🔬 [СРАВНЕНИЕ] Анализ топологии '{label1}' и '{label2}'...")
                    population.recall_from_hive_mind(label1)
                    population.recall_from_hive_mind(label2)
                    result = population.active.compare_entities(label1, label2, debug=True)
                    if 'error' in result:
                        print(f"   ❌ {result['error']}")
                        return
                    print(f"   🧬 Сравнение: '{result['entity1']}' vs '{result['entity2']}'")
                    if result['common']:
                        print(f"   🤝 Общие черты ({len(result['common'])}): {', '.join(result['common'])}")
                    else:
                        print(f'   🤝 Явных общих черт не обнаружено.')
                    if result['antagonisms']:
                        print(f'   ⚔️ АНТАГОНИЗМЫ:')
                        for idx, (a1, a2) in enumerate(result['antagonisms'], 1):
                            print(f"      {idx}. 🔴 {result['entity1']}: {a1}  <-->  {result['entity2']}: {a2}")
                            last_compare_antagonisms = result['antagonisms']
                            last_compare_all_antagonisms = result.get('all_antagonisms', [])
                            last_compare_entities = (label1, label2)
                        print(f"   💡 Команда 'убери N' — пометить антагонизм N как ложный")
                    else:
                        last_compare_antagonisms = []
                        print(f'   ⚪ Явных антагонизмов не найдено.')
                    print(f'   {population.active.calibration.get_feedback_hint()}')
                elif action == 'спроси':
                    if not args:
                        print('⚠️  Использование: спроси <вопрос>')
                        return
                    print(f"🔄 [АБДУКЦИЯ] Запуск обратной волны от вакуума '{args}'...")
                    query_label = membrane.resolve_query(args.strip())
                    query_hdc = population.active.encoder.encode(query_label)
                    if getattr(population.active, 'current_epoch', None):
                        query_hdc ^= population.active.epochs[population.active.current_epoch]
                    best_match = None
                    best_sim = -1.0
                    found_r = population.active.find_best_resonator_by_label(query_label)
                    if found_r:
                        best_match = found_r.id
                        best_sim = 1.0
                    else:
                        for label, r_id in population.active.label_to_id.items():
                            if label.startswith(('mod:', 'skill:', 'EPOCH:')) or label == 'SELF':
                                continue
                            r = population.active.resonators[r_id]
                            if not r.connections:
                                continue
                            sim = population.active.encoder.similarity(r.hdc_vector, query_hdc)
                            if sim > best_sim:
                                best_sim = sim
                                best_match = r_id
                    exact_id = None
                    if query_label in population.active.label_to_id:
                        exact_id = population.active.label_to_id[query_label]
                    elif f'root:{query_label}' in population.active.label_to_id:
                        exact_id = population.active.label_to_id[f'root:{query_label}']
                    if exact_id is not None:
                        best_match = exact_id
                        best_sim = 1.0
                    else:
                        for label, r_id in population.active.label_to_id.items():
                            if label.startswith(('mod:', 'skill:', 'EPOCH:')) or label == 'SELF':
                                continue
                            r = population.active.resonators[r_id]
                            if not r.connections:
                                continue
                            label_text = label.lower()
                            query_text = query_label.lower()
                            overlap = 1.0 if query_text in label_text or label_text in query_text else 0.0
                            sim = population.active.encoder.similarity(r.hdc_vector, query_hdc)
                            adjusted = sim + overlap * 0.15
                            if adjusted > best_sim:
                                best_sim = adjusted
                                best_match = r_id
                    if best_match is not None and best_sim > population.active.calibration.query_min_similarity:
                        anchor_label = population.active.resonators[best_match].label
                        print(f"   🎯 Ближайший якорь: '{anchor_label}' (схожесть: {best_sim:.3f})")
                        path = population.active.tick_backward(best_match, max_depth=6)
                    else:
                        vacuum_id = population.active.get_or_create(query_label).id
                        path = population.active.tick_backward(vacuum_id, max_depth=6)
                    if len(path) <= 1:
                        if population.recall_from_hive_mind(query_label):
                            found_r = population.active.find_best_resonator_by_label(query_label)
                            if found_r:
                                path = population.active.tick_backward(found_r.id, max_depth=6)
                            else:
                                vacuum_id = population.active.get_or_create(query_label).id
                                path = population.active.tick_backward(vacuum_id, max_depth=6)
                    path_labels = [population.active.resonators[pid].label for pid in path if pid in population.active.resonators]
                    print(f"   Трасса: {' <- '.join(path_labels)}")
                    population.active.last_query_path = path
                    population.active.last_query_label = query_label
                    population.active.inject_path_energy(path, energy=population.active.calibration.query_inject_energy)
                    print(f'   {population.active.calibration.get_feedback_hint()}')
                elif action == 'ответь':
                    print('🗣️ [ГЕНЕРАЦИЯ] Извлечение пути волны...')
                    if getattr(population.active, 'last_query_path', None):
                        path_labels = []
                        for node_id in population.active.last_query_path:
                            if node_id in population.active.resonators:
                                label = population.active.resonators[node_id].label
                                if label.startswith('root:'):
                                    clean = label[5:]
                                elif label.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')) or label == 'SELF':
                                    continue
                                else:
                                    clean = label
                                if len(clean) >= 3 and clean not in membrane.STOP_WORDS:
                                    path_labels.append(clean)
                        seen = set()
                        unique_labels = []
                        for lbl in path_labels:
                            if lbl not in seen:
                                seen.add(lbl)
                                unique_labels.append(lbl)
                        if unique_labels:
                            response = membrane.generate_text(unique_labels, max_length=8)
                            print(f'🗣️ КРИСТАЛЛ> {response}')
                        else:
                            print('⚪ Путь волны пуст.')
                        population.active.last_query_path = []
                    else:
                        active = population.active.get_active(5)
                        seed_concepts = [r.label for r in active if r.label != 'SELF' and (not r.label.startswith(('mod:', 'cluster:', 'mdl:', 'skill:', 'EPOCH:')))]
                        if seed_concepts:
                            response = membrane.generate_text(seed_concepts, max_length=8)
                            print(f'🗣️ КРИСТАЛЛ> {response}')
                        else:
                            print('⚪ Нет активных концептов для генерации.')
                    print(f'   {population.active.calibration.get_feedback_hint()}')
                elif action == 'читай':
                    if not args:
                        print('⚠️  Использование: читай <файл>')
                        return
                    print(f"📚 [ЭПОХА] Загрузка файла '{args}'...")
                    population.active.set_epoch(args)
                    try:
                        with open(args, 'r', encoding='utf-8') as f:
                            text = f.read()
                        stats = membrane.inject_text(text)
                        total_links = stats.cause_links + stats.except_links + stats.cond_links + stats.syntagm_links
                        population.active._last_read_count = total_links
                        print(f"   Эпоха '{args}' успешно вплавлена в биты памяти.")
                        print(f'   🔬 [СПЕКТР ТЕКСТА]:')
                        print(f'      ⚡ Причинных связей (CAUSE):  {stats.cause_links}')
                        print(f'      🛡 Исключений (EXCEPT):       {stats.except_links}')
                        print(f'      ❓ Условий (COND):            {stats.cond_links}')
                        print(f'      🔗 Синтагматика:              {stats.syntagm_links}')
                        print(f'      🆕 IS_A связей:               {stats.is_a_links}')
                        print(f'      🚫 Отрицаний (не):            {stats.negations}')
                        if population.check_population_dynamics():
                            membrane.lattice = population.active
                    except FileNotFoundError:
                        print(f'❌ Файл не найден: {args}')
                    except Exception as e:
                        print(f'❌ Ошибка чтения: {e}')
                elif action == 'вмешайся':
                    if not args:
                        print('⚠️ Использование: вмешайся <узел>')
                        return
                    print(f"🔬 [do(X)] Контрфактическое вмешательство в '{args}'...")
                    log = population.active.do_intervention(args, energy=population.active.calibration.intervention_energy)
                    for line in log:
                        print(f'  {line}')
                    for _ in range(3):
                        population.active.tick()
                    print(f'⚡ Результат после 3 тактов:')
                    print_status(population.active)
                elif action == 'аналогии':
                    if not args:
                        print('⚠️ Использование: аналогии <узел>')
                        return
                    results = population.active.find_analogies(args)
                    if results:
                        print(f"🔮 [АНАЛОГИИ] для '{args}':")
                        for label, weight in results:
                            print(f'  ≈ {label} (вес={weight})')
                    else:
                        print(f"⚪ Аналогий для '{args}' не найдено.")
                elif action == 'связи':
                    if not args:
                        print('⚠️ Использование: связи <узел>')
                        return
                    search_variants = [args, f'root:{args}', args.lower(), f'root:{args.lower()}']
                    found_r = None
                    for variant in search_variants:
                        if variant in population.active.label_to_id:
                            found_r = population.active.resonators[population.active.label_to_id[variant]]
                            break
                    if not found_r:
                        found_r = population.active.find_best_resonator_by_label(args)
                    if found_r:
                        print(f"🔗 [СВЯЗИ] '{found_r.label}' ({len(found_r.connections)} связей):")
                        for tgt_id, packed in sorted(found_r.connections.items(), key=lambda x: unpack_edge(x[1])[0], reverse=True):
                            if tgt_id in population.active.resonators:
                                w, et = unpack_edge(packed)
                                tname = EDGE_NAMES.get(et, '?')
                                tgt_label = population.active.resonators[tgt_id].label
                                print(f'  → {tgt_label} [{tname}] вес={w}')
                    else:
                        print(f"⚪ Узел '{args}' не найден.")
                elif action == 'план' or action == 'симулируй':
                    if not args:
                        print('⚠️  Использование: план <цель>')
                        return
                    print(f"🔮 [ПЛАНИРОВЩИК] Запуск теневой волны к цели '{args}'...")
                    population.recall_from_hive_mind(args)
                    history = population.active.plan(args, horizon=10)
                    for line in history:
                        print(line)
                    print(f'   {population.active.calibration.get_feedback_hint()}')
                elif action == 'популяция':
                    print(population.get_status())
                elif action in ('крути', 'настрой'):
                    parts_arg = args.split()
                    if len(parts_arg) != 2:
                        print('⚙️ Использование: крути <ось> <значение 0.0-1.0>')
                        print(population.active.calibration.get_axes_summary())
                        return
                    axis_name, val_str = parts_arg
                    try:
                        val = float(val_str)
                        alias_map = {'темперамент': 'temperament', 'огонь': 'temperament', 'лед': 'temperament', 'строгость': 'strictness', 'консерватизм': 'strictness', 'любопытство': 'curiosity', 'глубина': 'curiosity', 'речистость': 'verbality', 'поэт': 'verbality', 'телеграф': 'verbality'}
                        axis_key = alias_map.get(axis_name.lower(), axis_name.lower())
                        population.active.calibration.apply_macro_axis(axis_key, val)
                        population.active.calibration.auto_save()
                        if axis_key in ('strictness', 'curiosity'):
                            membrane.invalidate_cache()
                    except ValueError:
                        print('⚠️ Значение должно быть числом.')
                elif action in ('режим', 'пресет'):
                    population.active.calibration.apply_preset(args.strip())
                    membrane.invalidate_cache()
                elif action == 'калибруй':
                    if not args:
                        print('⚙️ [КАЛИБРОВКА] Текущее состояние осей:')
                        print(population.active.calibration.get_axes_summary())
                        print('\n⚙️ Сырые параметры (топ-10):')
                        cal = population.active.calibration
                        print(f'   energy_cap={cal.energy_cap}, base_decay={cal.base_decay}')
                        print(f'   query_min_sim={cal.query_min_similarity:.2f}, speech_filter={cal.speech_semantic_filter:.2f}')
                        print(f'   jaccard_thresh={cal.syngrammy_jaccard_thresh:.2f}, mdl_triads={cal.mdl_triad_count_thresh}')
                        print(f'   backward_depth={cal.backward_max_depth}, inject={cal.concept_inject_energy}')
                    else:
                        try:
                            key, val = args.split()
                            if hasattr(population.active.calibration, key):
                                field_type = type(getattr(population.active.calibration, key))
                                setattr(population.active.calibration, key, field_type(val))
                                population.active.calibration.auto_save()
                                print(f"✅ Порог '{key}' изменен на {val}")
                            else:
                                print(f'❌ Неизвестный параметр: {key}')
                        except ValueError:
                            print('⚠️ Использование: калибруй <параметр> <значение>')
                else:
                    print(f'❓ Неизвестная команда: {action}')
            except KeyboardInterrupt:
                print('\n💎 Прервано пользователем.')
                return False
            except Exception as e:
                raise
        finally:
            self.last_state_cache = last_state_cache
            self.last_compare_all_antagonisms = last_compare_all_antagonisms
            self.last_compare_antagonisms = last_compare_antagonisms
            self.last_compare_entities = last_compare_entities
