# Аудит Crystal2

Аудит исходного HEAD feeac6f; изменения выполняются поэтапно, без замены архитектуры.

| Приоритет | Подтвержденный дефект | Место и воспроизведение |
|---|---|---|
| P0 | Корреляция превращается в причинность без свидетельств; параметр min_co_activation не используется | `CrystalLattice.infer_causal_links`: активная пара с SYNTAGM > 50 |
| P0 | Число соседей автоматически подтверждает гипотезу | `discover_abstractions`, `defragment`: support >= 3 и count >= 5 |
| P0 | Побежденные убеждения больше не сравниваются при изменении надежности источника | `BeliefRevisionEngine._resolve_conflicts`, `set_source_reliability` |
| P0 | Равный вес противоречий разрешается временем поступления; unknown может победить содержательный факт | `BeliefRevisionEngine.query`, `_resolve_conflicts` |
| P0 | Более слабое возражение заменяет статус сильного убеждения на contradiction; старое основание перезаписывается | `CrystalLattice.resolve_belief_conflict` |
| P1 | MDL gain сокращается до (support-1)*len(pattern)-1 независимо от качества следствия | `ConceptFormationEngine._evaluate_pattern` |
| P1 | Повторный анализ накапливает дубликаты, бинарные предикаты превращаются в унарные правила, контексты смешиваются | `ConceptFormationEngine.analyze_facts`, `_extract_predicate`, `auto_generalize` |
| P1 | Четный bundle выбирает ноль при равенстве голосов (для пары плотность ~25%) | `HDCEncoder.bundle`; encode SHA256 воспроизводим, XOR обратим; сходство строк не означает семантического сходства |
| P1 | Причинный маркер трактуется как факт, отрицание относится к узлу вместо отношения | `LanguageMembrane._process_sentence`: разбор текста не является проверкой истинности |
| P2 | Снимки сохраняют граф/энергию, но не hypotheses, reasons, belief_context и историю; загрузка в существующую lattice может оставить устаревшие hypotheses | `CrystalSnapshot.save`, `_load_full`, `save_delta` |
| P2 | Отслеживаются pyc; нет README и независимого benchmark | дерево Git |

План проверки: регрессии P0, lifecycle гипотез с независимыми свидетельствами,
синтетические эксперименты A–D, HDC-инварианты, legacy snapshot roundtrip.
Наличие новых узлов/ребер не является метрикой обучения.
