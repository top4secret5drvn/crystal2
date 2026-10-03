"""
calibration.py — Профили калибровки и RLHF (Ц7).
Управляет сырыми параметрами Кристалла через макро-оси и обратную связь.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict

CONFIG_PATH = os.path.join(os.getcwd(), 'crystal_calibration.json')

@dataclass
class CalibrationProfile:
    # === Сырые параметры (Энергия и Затухание) ===
    energy_cap: int = 3000
    base_decay: int = 25
    concept_inject_energy: int = 800
    intervention_energy: int = 500
    shadow_energy_cap: int = 2000
    
    # === Морфология и Синтаксис ===
    min_root_len: int = 3
    lcs_search_limit: int = 1000
    syntagm_weight_direct: int = 20
    syntagm_weight_skip: int = 10
    causal_marker_weight: int = 50
    except_antonym_weight: int = 80
    
    # === Генерация речи ===
    speech_min_word_len: int = 3
    speech_semantic_filter: float = 0.40
    
    # === Сон и Абстракции (Сингамия / MDL) ===
    syngrammy_min_attractor_energy: int = 50
    syngrammy_min_edge_weight: int = 20
    syngrammy_min_profile_len: int = 3
    syngrammy_jaccard_thresh: float = 0.45
    mdl_triad_min_weight: int = 30
    mdl_triad_min_energy: int = 30
    mdl_triad_count_thresh: int = 2
    
    # === Обратная волна (Tick Backward) ===
    backward_max_depth: int = 7
    backward_cause_mult: float = 2.0
    backward_effect_mult: float = 1.5
    backward_cond_mult: float = 1.2
    backward_syn_mult: float = 1.0
    backward_true_state_bonus: int = 50
    backward_abstract_bonus: int = 30
    backward_vacuum_anchor_sim: float = 0.50
    
    # === Запросы и Поиск ===
    query_min_similarity: float = 0.40
    query_inject_energy: int = 200
    
    # === Макро-оси (0.0 - 1.0) ===
    axis_temperament: float = 0.50  # Огонь / Лед
    axis_strictness: float = 0.50   # Строгость / Консерватизм
    axis_curiosity: float = 0.50    # Любопытство / Глубина
    axis_verbality: float = 0.50    # Поэт / Телеграф

    # ============================================================
    # Загрузка и Сохранение
    # ============================================================
    @classmethod
    def auto_load(cls) -> 'CalibrationProfile':
        """Загружает профиль из JSON, если он существует."""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                profile = cls()
                for k, v in data.items():
                    if hasattr(profile, k):
                        # Приведение типа для безопасности
                        field_type = type(getattr(profile, k))
                        setattr(profile, k, field_type(v))
                print(f"⚙️ Загружен профиль калибровки: {CONFIG_PATH}")
                return profile
            except Exception as e:
                print(f"⚠️ Ошибка загрузки калибровки: {e}. Используется профиль по умолчанию.")
        return cls()

    def auto_save(self):
        """Сохраняет текущие параметры в JSON."""
        try:
            data = {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, str, bool))}
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Не удалось сохранить калибровку: {e}")

    # ============================================================
    # Макро-оси и Пресеты
    # ============================================================
    def apply_macro_axis(self, axis: str, value: float, silent: bool = False):
        """Применяет значение макро-оси к сырым параметрам."""
        value = max(0.0, min(1.0, float(value)))
        
        if axis == "temperament":
            self.axis_temperament = value
            self.energy_cap = int(3000 + value * 4000)
            self.concept_inject_energy = int(100 + value * 200)
        elif axis == "strictness":
            self.axis_strictness = value
            self.min_root_len = int(3 + (1.0 - value) * 3)  # 6..3
            self.query_min_similarity = 0.3 + value * 0.4    # 0.3..0.7
            self.speech_semantic_filter = 0.2 + value * 0.6  # 0.2..0.8
        elif axis == "curiosity":
            self.axis_curiosity = value
            self.backward_max_depth = int(4 + value * 8)     # 4..12
            self.syngrammy_jaccard_thresh = 0.6 - value * 0.3 # 0.6..0.3
            self.mdl_triad_count_thresh = max(2, int(5 - value * 3)) # 5..2
        elif axis == "verbality":
            self.axis_verbality = value
            self.speech_min_word_len = max(2, int(4 - value * 2)) # 4..2
        else:
            if not silent:
                print(f"❓ Неизвестная ось: {axis}")
            return
            
        if not silent:
            print(f"⚙️ Ось '{axis}' установлена в {value:.2f}")

    def apply_preset(self, preset: str):
        """Применяет готовый пресет поведения."""
        presets = {
            "логик":    {"temperament": 0.2, "strictness": 0.9, "curiosity": 0.5, "verbality": 0.3},
            "поэт":     {"temperament": 0.8, "strictness": 0.2, "curiosity": 0.7, "verbality": 0.9},
            "параноик": {"temperament": 0.4, "strictness": 0.9, "curiosity": 0.9, "verbality": 0.4},
            "ребенок":  {"temperament": 0.9, "strictness": 0.1, "curiosity": 0.9, "verbality": 0.8},
            "ученый":   {"temperament": 0.4, "strictness": 0.7, "curiosity": 0.9, "verbality": 0.6},
        }
        if preset not in presets:
            print(f"❌ Неизвестный пресет '{preset}'. Доступны: {', '.join(presets.keys())}")
            return
        print(f"🎛️ Применение пресета: {preset.upper()}")
        for axis, val in presets[preset].items():
            self.apply_macro_axis(axis, val, silent=True)
        self.auto_save()

    def get_axes_summary(self) -> str:
        """Возвращает текстовое представление текущих осей."""
        return (
            f"   🎛️ Макро-оси:\n"
            f"      Темперамент (огонь/лед):   {self.axis_temperament:.2f}\n"
            f"      Строгость (консерватизм):  {self.axis_strictness:.2f}\n"
            f"      Любопытство (глубина):     {self.axis_curiosity:.2f}\n"
            f"      Речистость (поэт/телеграф):{self.axis_verbality:.2f}"
        )

    # ============================================================
    # RLHF (Обучение на основе отзывов пользователя)
    # ============================================================
    def feedback(self, complaint: str) -> bool:
        """Обрабатывает жалобу пользователя и подкручивает параметры."""
        if complaint == "бессвязно":
            self.speech_semantic_filter = min(0.95, self.speech_semantic_filter + 0.1)
            print("   🔧 Увеличен порог семантической связности.")
        elif complaint == "мимо":
            self.query_min_similarity = min(0.90, self.query_min_similarity + 0.05)
            print("   🔧 Увеличена строгость поиска якоря.")
        elif complaint == "кратко":
            self.axis_verbality = min(1.0, self.axis_verbality + 0.1)
            self.apply_macro_axis("verbality", self.axis_verbality, silent=True)
            print("   🔧 Увеличена речистость.")
        elif complaint == "длинно":
            self.axis_verbality = max(0.0, self.axis_verbality - 0.1)
            self.apply_macro_axis("verbality", self.axis_verbality, silent=True)
            print("   🔧 Уменьшена речистость.")
        elif complaint == "сухо":
            self.axis_temperament = min(1.0, self.axis_temperament + 0.1)
            self.apply_macro_axis("temperament", self.axis_temperament, silent=True)
            print("   🔧 Добавлено больше 'огня'.")
        elif complaint == "бред":
            self.axis_strictness = min(1.0, self.axis_strictness + 0.1)
            self.apply_macro_axis("strictness", self.axis_strictness, silent=True)
            print("   🔧 Увеличена строгость логики.")
        elif complaint == "ok":
            print("   ✅ Ответ принят, параметры не изменены.")
        else:
            return False
            
        self.auto_save()
        return True

    def get_feedback_hint(self) -> str:
        """Подсказка для пользователя после генерации ответа."""
        return "💡 Оцените ответ: !бессвязно !мимо !кратко !длинно !сухо !бред !ok"