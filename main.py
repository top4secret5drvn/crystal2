"""
main.py — CLI-интерфейс Кристалла v7.1.
Объединение Векторов 1-6 + Митоз + Калибровка + RLHF + Вердикт-модули.
"""
import traceback
from engine import (CrystalLattice, CrystalPopulation, pack_edge, unpack_edge, EDGE_NAMES,
                    EDGE_CAUSE, EDGE_EXCEPT, EDGE_ANALOG, EDGE_GOAL, CrystalReason)
from membrane import LanguageMembrane
from persistence import CrystalSnapshot
from calibration import CalibrationProfile
import urllib.request
import urllib.parse
import json
import os


def print_status(lattice: CrystalLattice):
    """Визуализация состояния с поддержкой XAI, Любопытства, Парадоксов и Субъектности."""
    print("\n" + "=" * 60)
    print(f"🧠 КРИСТАЛЛ | Такт: {lattice.tick_count} | Узлов: {len(lattice.resonators)}")
    if getattr(lattice, 'current_epoch', None):
        print(f"📚 ТЕКУЩАЯ ЭПОХА: '{lattice.current_epoch}' | Навыков (Skills): {len(lattice.skills)}")
    # 🆕 Подсказка о новых командах
    n_hyp = len(getattr(lattice, 'hypotheses', {}))
    if n_hyp > 0:
        print(f"🧪 ГИПОТЕЗ: {n_hyp} активных (команда: 'гипотезы')")
    n_ctx = len(getattr(lattice, 'contexts', {}))
    if n_ctx > 1:
        print(f"🗂️  КОНТЕКСТОВ: {n_ctx} (команда: 'контексты')")
    paradox_ids = lattice.detect_paradoxes_bitwise()
    if paradox_ids:
        print("\n🚨 КОГНИТИВНЫЙ ДИССОНАНС (Парадоксы):")
        for pid in paradox_ids:
            if pid in lattice.resonators:
                print(f"   ⚠️  Узел '{lattice.resonators[pid].label}' находится в состоянии ПАРАДОКС!")
        print("   " + "-" * 56)
    if hasattr(lattice, 'interference_log') and lattice.interference_log:
        print("\n✨ ОЗАРЕНИЯ (Интерференция волн):")
        for insight in lattice.interference_log[-3:]:
            print(f"   ⚡ {insight}")
        print("   " + "-" * 56)
    if hasattr(lattice, 'paradox_log') and lattice.paradox_log:
        print("\n📜 ПОСЛЕДНИЕ ЛОГИЧЕСКИЕ ОПЕРАЦИИ:")
        for log in lattice.paradox_log[-3:]:
            print(f"   {log}")
        print("   " + "-" * 56)
    if hasattr(lattice, 'get_questions'):
        questions = lattice.get_questions(3)
        valid_questions = [(r, e) for r, e in questions if e > 0.8]
        if valid_questions:
            print("\n❓ ВОПРОСЫ СИСТЕМЫ (Зоны информационного вакуума):")
            for r, entropy in valid_questions:
                conn_count = len(r.connections)
                print(f"   🧐 '{r.label}' (Энтропия: {entropy:.2f}, Связей: {conn_count}) -> Требует контекста.")
            print("   " + "-" * 56)
            print("   💡 [ДЕМОН ЛЮБОПИТСТВА] Обнаружен вакуум. Чтобы система САМА нашла определение")
            print("      в Wikipedia, проанализировала и сохранила, введите:")
            top_q = valid_questions[0][0].label
            clean_q = top_q.replace('root:', '').replace('cluster:', '')
            print(f"      👉 изучи {clean_q}")
    active = lattice.get_active(7)
    if not active:
        print("\n⚪  Активных резонаторов нет (тишина).")
    else:
        print("\n🔥 Топ активных резонаторов:")
        print(f"   {'Энергия':<10} {'Статус':<12} {'Увер.':<8} {'Метка':<20} {'Посл. такт':<10}")
        print("   " + "-" * 60)
        for r in active:
            state_str = r.state.name
            belief = r.belief_status[:10] if hasattr(r, 'belief_status') else '?'
            conf = f"{r.belief_confidence:.1f}" if hasattr(r, 'belief_confidence') else "?"
            print(f"   [{r.energy:<8}] {state_str:<12} {conf:<8} {r.label:<20} {r.last_tick:<10}")
    print("=" * 60 + "\n")


def wiki_probe(query: str) -> str:
    """Compatibility wrapper around the bounded external acquisition layer."""
    from research import fetch_wikipedia
    try:
        return fetch_wikipedia(query)['text']
    except (OSError, ValueError) as error:
        print(f"Wikipedia: {error}")
        return None


def autonomous_research(membrane, lattice, query: str, ticks: int = 15):
    from research import research_steps
    safe_name = "".join(c for c in query if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
    try:
        for event in research_steps(membrane,lattice,query,ticks,snapshot_path=f"auto_wiki_{safe_name}.cry"):
            print(f"[ИЗУЧИ] {event}")
    except (OSError, ValueError) as error:
        print(f"[ИЗУЧИ] Остановлено: {error}")


def auto_load_crystals(population: CrystalPopulation, membrane: LanguageMembrane) -> int:
    import glob
    cry_files = glob.glob("*.cry")
    if not cry_files:
        return 0
    cry_files.sort(key=lambda f: os.path.getmtime(f))
    print(f"\n🔄 [АВТОЗАГРУЗКА] Обнаружено {len(cry_files)} снапшотов памяти:")
    for f in cry_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"   📦 {f} ({size_kb:.1f} KB)")
    current_calibration = population.active.calibration
    population.crystals.clear()
    population.active_index = 0
    loaded_crystals = []
    loaded_filenames = []
    for filename in cry_files:
        crystal = CrystalLattice()
        crystal.calibration = current_calibration
        if CrystalSnapshot.load(crystal, filename):
            loaded_crystals.append(crystal)
            loaded_filenames.append(filename)
    if not loaded_crystals:
        population.spawn_initial()
        population.active.calibration = current_calibration
        membrane.lattice = population.active
        return 0
    population.crystals = loaded_crystals
    population.active_index = len(population.crystals) - 1
    for i, crystal in enumerate(population.crystals):
        fname = loaded_filenames[i]
        if i != population.active_index:
            crystal.enter_dormancy()
            print(f"   💤 Кристалл #{i+1} ({fname}) → Проводник (долговременная память)")
        else:
            print(f"   🔥 Кристалл #{i+1} ({fname}) → АКТИВНЫЙ (текущее сознание)")
    membrane.lattice = population.active
    print(f"\n✅ [АВТОЗАГРУЗКА] Загружено {len(loaded_crystals)} кристаллов. "
          f"Активен: #{population.active_index + 1}, "
          f"Спящих (подсознание): {len(loaded_crystals) - 1}")
    return len(loaded_crystals)


def main():
    print("✨ КРИСТАЛЛ v7.1 (Вердикт + Калибровка + RLHF + Митоз + Полный спектр) ✨")
    print("Команды Цикла Наблюдателя и Диалога:")
    print("  ввод   <текст>          — Впустить данные (Факт)")
    print("  ложь   <текст>          — Опровергнуть факт (Ложь)")
    print("  отрицание <А> <Б>       — А опровергает Б (Defeater)")
    print("  думай [N]              — Запустить N тактов мышления")
    print("  состояние              — Показать срез памяти")
    print("  сохрани <имя>          — Базовый снапшот (.cry)")
    print("  эпизод  <имя>          — Дельта-снапшот (.cdt)")
    print("  загрузи <имя>          — Восстановление из гибернации")
    print("  сон                    — Дефрагментация и Отжиг")
    print("  отжиг                  — Ручной термический отжиг")
    print("  спроси  <вопрос>       — 🚀 Задать вопрос (Abductive Ping)")
    print("  ответь                 — 🚀 Сгенерировать ответ")
    print("  объясни                — 🧭 XAI-отчёт для последнего запроса")
    print("  читай   <файл>         — 🚀 Загрузить книгу")
    print("  вмешайся <узел>        — 🔬 do(X) Контрфактическое вмешательство")
    print("  аналогии <узел>        — 🔮 Поиск аналогий")
    print("  связи   <узел>         — 🔗 Показать связи узла")
    print("  план    <цель>         — 🔮 Симулировать путь к цели")
    print("  сравни  <А> от <Б>     — 🔬 Сравнить два концепта")
    print("  популяция              — 🧬 Статус популяции")
    print("\n🆕 ВЕРДИКТ-КОМАНДЫ:")
    print("  гипотезы               — 🧪 Показать активные гипотезы")
    print("  подтверди <узел>       — ✅ Подтвердить гипотезу")
    print("  опровергни <узел>      — ❌ Отклонить гипотезу")
    print("  контексты              — 🗂️  Показать иерархию контекстов")
    print("  качество <цель>        — 📊 Оценить план без выполнения")
    print("\n🎛️ КОМАНДЫ КАЛИБРОВКИ:")
    print("  крути   <ось> <0.0-1.0> — Повернуть макро-ось")
    print("  режим   <пресет>        — Применить профиль")
    print("  калибруй                — Показать состояние осей")
    print("  ! <жалоба>              — RLHF: оценка ответа")
    print("  убери   <N>             — 🔬 Пометить антагонизм N как ложный")
    print("  мысль   <концепт>       — 🧠 Процесс мышления")
    print("  фреймы                  — 🎭 Показать активные фреймы")
    print("  выход                   — Завершить сеанс\n")

    population = CrystalPopulation()
    population.active.calibration = CalibrationProfile.auto_load()
    for axis in ["temperament", "strictness", "curiosity", "verbality"]:
        val = getattr(population.active.calibration, f"axis_{axis}")
        population.active.calibration.apply_macro_axis(axis, val, silent=True)
    membrane = LanguageMembrane(population.active)
    auto_load_crystals(population, membrane)

    from command_session import CommandSession
    session = CommandSession(population, membrane)
    while True:
        try:
            command = input('КРИСТАЛЛ> ')
        except (EOFError, KeyboardInterrupt):
            break
        result = session.execute(command, capture=False)
        print(result.output, end='')
        if not result.keep_running:
            break


if __name__ == "__main__":
    main()