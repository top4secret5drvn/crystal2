"""
persistence.py — Бинарные снапшоты и Дельта-кодирование (Вектор 4).
"""
import struct
import time
import zlib
from pathlib import Path

class CrystalSnapshot:
    MAGIC_FULL = b'CRYSTAL\x01'
    MAGIC_DELTA = b'CRYSTAL\x02'
    # 🆕 Шаг 0.5 (исправление): явный признак формата v2 — резонаторные рекорды
    # содержат SemanticVector после HDC. Эвристический probe по содержимому
    # давал ложные срабатывания; версия кодируется в magic-сигнатуре файла.
    MAGIC_FULL_V2 = b'CRYSTAL\x03'
    HEADER_FORMAT = '<8s Q Q I I I I'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    BYTES_PER_VECTOR = 1250
    # 🆕 ФАЗА 0, Шаг 0.5: версия формата записи резонатора
    #   v1 (старые файлы): <I H {lbl_len}s 1250s B I Q          — без семантического вектора
    #   v2 (новые файлы) : <I H {lbl_len}s 1250s H {sem_len}s B I Q — после HDC идут байты SemanticVector
    RES_RECORD_VERSION_V2 = 2

    @staticmethod
    def _pack_resonator_record(id_: int, lbl_b: bytes, hdc: bytes, sem_bytes: bytes, state: int, energy: int, last_tick: int) -> bytes:
        """Запись полного рекорда резонатора (v2: HDC + SemanticVector)."""
        return struct.pack(f'<I H {len(lbl_b)}s 1250s H {len(sem_bytes)}s B I Q',
                           id_, len(lbl_b), lbl_b, hdc,
                           len(sem_bytes), sem_bytes,  # ← НОВОЕ (Шаг 0.5)
                           state, energy, last_tick)
    
    @staticmethod
    def rle_compress(data: bytes) -> bytes:
        """PackBits RLE: Сжимает длинные последовательности нулей в 2 байта."""
        if not data: return b''
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            run_len = 1
            while i + run_len < n and data[i + run_len] == data[i] and run_len < 128:
                run_len += 1
            if run_len >= 3:
                out.append(128 + run_len - 1)
                out.append(data[i])
                i += run_len
            else:
                lit_len = 0
                while i + lit_len < n and lit_len < 128:
                    next_run = 1
                    while i + lit_len + next_run < n and data[i + lit_len + next_run] == data[i + lit_len] and next_run < 3:
                        next_run += 1
                    if next_run >= 3: break
                    lit_len += 1
                if lit_len == 0: lit_len = 1
                out.append(lit_len - 1)
                out.extend(data[i:i+lit_len])
                i += lit_len
        return bytes(out)
    
    @staticmethod
    def rle_decompress(data: bytes) -> bytes:
        """Распаковка PackBits RLE."""
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            header = data[i]
            i += 1
            if header >= 128:
                count = header - 128 + 1
                val = data[i]
                i += 1
                out.extend([val] * count)
            else:
                count = header + 1
                out.extend(data[i:i+count])
                i += count
        return bytes(out)
    
    @staticmethod
    def generate_state_cache(lattice) -> dict:
        """Создает снимок состояния в памяти для последующего XOR."""
        cache = {}
        for r in lattice.resonators.values():
            cache[r.id] = {
                'state': int(r.state),
                'energy': r.energy,
                'tick': r.last_tick,
                'conns': dict(r.connections),
                'defeats': list(r.defeats),
                'ctx': list(r.context_sources)
            }
        return cache
    
    @staticmethod
    def save(lattice, filename: str):
        """Сохраняет ПОЛНЫЙ базовый снапшот (.cry)."""
        timestamp = int(time.time())
        tick_count = lattice.tick_count
        res_count = len(lattice.resonators)
        connections, defeaters, contexts = [], [], []
        for r in lattice.resonators.values():
            for t, w in r.connections.items(): connections.append((r.id, t, w))
            for t in r.defeats: defeaters.append((r.id, t))
            if r.context_sources: contexts.append((r.id, r.context_sources))
        
        buffer = bytearray()
        header = struct.pack(
            CrystalSnapshot.HEADER_FORMAT,
            CrystalSnapshot.MAGIC_FULL_V2, timestamp, tick_count, res_count,
            len(connections), len(contexts), len(defeaters)
        )
        buffer.extend(header)
        lattice._snapshot_format = 2  # 🆕 дельты после этого снапшота наследуют v2
        
        for r in lattice.resonators.values():
            lbl = r.label.encode('utf-8')
            hdc = r.hdc_vector.to_bytes(CrystalSnapshot.BYTES_PER_VECTOR, 'little')
            sem_bytes = r.semantic.to_bytes()  # 🆕 Шаг 0.5: SemanticVector после HDC
            buffer.extend(CrystalSnapshot._pack_resonator_record(
                r.id, lbl, hdc, sem_bytes, int(r.state), r.energy, r.last_tick))
        
        for s, t, w in connections: buffer.extend(struct.pack('<I I I', s, t, w))
        for s, t in defeaters: buffer.extend(struct.pack('<I I', s, t))
        
        for rid, srcs in contexts:
            buffer.extend(struct.pack('<I H', rid, len(srcs)))
            for s in srcs: buffer.extend(s.to_bytes(CrystalSnapshot.BYTES_PER_VECTOR, 'little'))
        
        crc = zlib.crc32(buffer) & 0xFFFFFFFF
        buffer.extend(struct.pack('<I', crc))
        
        with open(filename, 'wb') as f: f.write(buffer)
        print(f"[ПАМЯТЬ] ⚡ Базовый снапшот сохранён: {filename} ({len(buffer)} байт)")
    
    @staticmethod
    def save_delta(lattice, filename: str, prev_cache: dict):
        """🚀 Вектор 4: Сохраняет ТОЛЬКО ИЗМЕНЕНИЯ (XOR + RLE)."""
        buffer = bytearray()
        # 🆕 Шаг 0.5: дельта наследует версию базового снапшота — v2-флаг
        # кладётся в заголовок патча, чтобы _load_delta знал формат рекордов.
        is_v2 = getattr(lattice, '_snapshot_format', 1) >= 2
        buffer.extend(CrystalSnapshot.MAGIC_DELTA)
        
        curr_cache = CrystalSnapshot.generate_state_cache(lattice)
        deleted = [pid for pid in prev_cache if pid not in curr_cache]
        new_ids = [cid for cid in curr_cache if cid not in prev_cache]
        updated = [cid for cid in curr_cache if cid in prev_cache and (
            curr_cache[cid]['state'] != prev_cache[cid]['state'] or
            curr_cache[cid]['energy'] != prev_cache[cid]['energy'] or
            curr_cache[cid]['tick'] != prev_cache[cid]['tick'] or
            curr_cache[cid]['conns'] != prev_cache[cid]['conns'] or
            curr_cache[cid]['ctx'] != prev_cache[cid]['ctx']
        )]
        
        # Заголовок патча: три счётчика + байт версии формата рекордов
        buffer.extend(struct.pack('<I I I B', len(deleted), len(new_ids), len(updated),
                                  2 if is_v2 else 1))
        
        for pid in deleted: buffer.extend(struct.pack('<I', pid))
        
        for nid in new_ids:
            r = lattice.resonators[nid]
            lbl = r.label.encode('utf-8')
            hdc = r.hdc_vector.to_bytes(CrystalSnapshot.BYTES_PER_VECTOR, 'little')
            sem_bytes = r.semantic.to_bytes()  # 🆕 Шаг 0.5: SemanticVector после HDC
            buffer.extend(CrystalSnapshot._pack_resonator_record(
                r.id, lbl, hdc, sem_bytes, int(r.state), r.energy, r.last_tick))
            for t, w in r.connections.items(): buffer.extend(struct.pack('<I I', t, w))
            buffer.extend(struct.pack('<I', 0xFFFFFFFF))
            buffer.extend(struct.pack('<H', len(r.context_sources)))
            for s in r.context_sources: buffer.extend(s.to_bytes(CrystalSnapshot.BYTES_PER_VECTOR, 'little'))
        
        for uid in updated:
            r = lattice.resonators[uid]
            p = prev_cache[uid]
            d_state = int(r.state) ^ p['state']
            d_energy = r.energy ^ p['energy']
            d_tick = r.last_tick ^ p['tick']
            buffer.extend(struct.pack('<I B I Q', uid, d_state, d_energy, d_tick))
            if r.connections != p['conns']:
                for t, w in r.connections.items(): buffer.extend(struct.pack('<I I', t, w))
            buffer.extend(struct.pack('<I', 0xFFFFFFFF))
            buffer.extend(struct.pack('<H', len(r.context_sources)))
            for s in r.context_sources: buffer.extend(s.to_bytes(CrystalSnapshot.BYTES_PER_VECTOR, 'little'))
        
        # 🚀 ИСПРАВЛЕНО: Сжимаем ВСЁ после MAGIC (с 8-го байта), а не с 12-го
        compressed_payload = CrystalSnapshot.rle_compress(bytes(buffer[8:]))
        final_buffer = bytearray(CrystalSnapshot.MAGIC_DELTA)
        final_buffer.extend(struct.pack('<I', len(buffer) - 8))  # Original size
        final_buffer.extend(compressed_payload)
        
        crc = zlib.crc32(final_buffer) & 0xFFFFFFFF
        final_buffer.extend(struct.pack('<I', crc))
        
        with open(filename, 'wb') as f: f.write(final_buffer)
        print(f"[ПАМЯТЬ] ⚡ Эпизод (Дельта) сохранён: {filename} ({len(final_buffer)} байт | Сжатие RLE)")
    
    @staticmethod
    def load(lattice, filename: str, prev_cache: dict = None) -> bool:
        path = Path(filename)
        if not path.exists():
            print(f"[ПАМЯТЬ] Файл не найден: {filename}")
            return False
        
        with open(path, 'rb') as f: buffer = f.read()
        magic = buffer[:8]
        
        if magic == CrystalSnapshot.MAGIC_DELTA:
            if prev_cache is None:
                print("[ПАМЯТЬ] Ошибка: Для загрузки дельты (.cdt) нужен базовый снапшот в памяти!")
                return False
            return CrystalSnapshot._load_delta(lattice, buffer, prev_cache)
        elif magic in (CrystalSnapshot.MAGIC_FULL, CrystalSnapshot.MAGIC_FULL_V2):
            # MAGIC_FULL (\x01) — legacy v1 без SemanticVector;
            # MAGIC_FULL_V2 (\x03) — v2 с SemanticVector. Версия кодируется
            # в сигнатуре, эвристический probe по содержимому больше не нужен.
            return CrystalSnapshot._load_full(lattice, buffer)
        else:
            print("[ПАМЯТЬ] Ошибка: Неизвестный формат файла!")
            return False
    
    @staticmethod
    def _load_full(lattice, buffer: bytes) -> bool:
        stored_crc = struct.unpack('<I', buffer[-4:])[0]
        if zlib.crc32(buffer[:-4]) & 0xFFFFFFFF != stored_crc:
            print("[ПАМЯТЬ] Ошибка CRC32 полного снапшота!")
            return False
        
        magic, timestamp, tick_count, res_count, conn_count, ctx_count, def_count = struct.unpack_from(CrystalSnapshot.HEADER_FORMAT, buffer, 0)
        # 🆕 Шаг 0.5 (исправление): версия формата определяется ЯВНО по magic,
        # а не эвристическим probe по содержимому (давало ложные срабатывания).
        is_v2 = (magic == CrystalSnapshot.MAGIC_FULL_V2)
        
        lattice.resonators.clear()
        lattice.label_to_id.clear()
        lattice.tick_count = tick_count
        if hasattr(lattice, 'syn_graph'):
            lattice.syn_graph.clear()
        
        offset = CrystalSnapshot.HEADER_SIZE
        max_id = 0
        
        from engine import Resonator, TruthValue, SemanticVector, NodeType
        
        for _ in range(res_count):
            id_, lbl_len = struct.unpack_from('<I H', buffer, offset)
            if is_v2:
                sem_len, = struct.unpack_from('<H', buffer,
                                              offset + 6 + lbl_len + CrystalSnapshot.BYTES_PER_VECTOR)
                # 🚀 ИСПРАВЛЕНО: формат '<I H {l}s 1250s H {n}s B I Q' даёт 9 значений
                # (второй H — это sem_len), а не 8 — отсюда был "too many values to unpack".
                fmt = f'<I H {lbl_len}s 1250s H {sem_len}s B I Q'
                (_rid, _ll, lbl_b, hdc_b, _sl, _sb, st, en, tk) = struct.unpack_from(fmt, buffer, offset)
                probe_off = offset + 6 + lbl_len + CrystalSnapshot.BYTES_PER_VECTOR
                sem, _ = SemanticVector.from_bytes(buffer, probe_off + 2)
            else:
                fmt = f'<I H {lbl_len}s 1250s B I Q'
                id_, _, lbl_b, hdc_b, st, en, tk = struct.unpack_from(fmt, buffer, offset)
                sem = SemanticVector()
            offset += struct.calcsize(fmt)
            
            lbl = lbl_b.decode('utf-8')
            r = Resonator(id=id_, label=lbl, hdc_vector=int.from_bytes(hdc_b, 'little'), energy=en, last_tick=tk)
            # Восстановление типа узла по префиксу метки (ФАЗА 1)
            if lbl.startswith('tok:'):   r.node_type = NodeType.TOKEN
            elif lbl.startswith('lem:'): r.node_type = NodeType.LEMMA
            elif lbl.startswith('op:'):  r.node_type = NodeType.OPERATOR
            elif lbl.startswith('sense:'): r.node_type = NodeType.SENSE
            elif lbl.startswith('ent:'): r.node_type = NodeType.ENTITY
            r.semantic = sem
            r.state = TruthValue(st)
            lattice.resonators[id_] = r
            lattice.label_to_id[lbl] = id_
            
            if id_ > max_id: max_id = id_
        
        for _ in range(conn_count):
            s, t, w = struct.unpack_from('<I I I', buffer, offset)
            offset += 12
            if s in lattice.resonators:
                lattice.resonators[s].connections[t] = w
        
        for _ in range(def_count):
            s, t = struct.unpack_from('<I I', buffer, offset)
            offset += 8
            if s in lattice.resonators:
                lattice.resonators[s].defeats.append(t)
        
        for _ in range(ctx_count):
            rid, src_len = struct.unpack_from('<I H', buffer, offset)
            offset += 6
            if rid in lattice.resonators:
                r = lattice.resonators[rid]
                for __ in range(src_len):
                    hdc_b = buffer[offset:offset+1250]
                    offset += 1250
                    r.context_sources.append(int.from_bytes(hdc_b, 'little'))
        
        lattice._next_id = max_id + 1 if lattice.resonators else 0
        
        if not hasattr(lattice, 'interference_log'): lattice.interference_log = []
        if not hasattr(lattice, 'paradox_log'): lattice.paradox_log = []
        
        print(f"[ПАМЯТЬ] ⚡ Базовый снапшот загружен: {len(lattice.resonators)} узлов.")
        return True
    
    @staticmethod
    def _delta_header_size(payload: bytes):
        """
        🆕 Шаг 0.5 (исправление): определение заголовка патча дельты.
        Новый формат (v2+): <I I I B> = 13 байт (3 счётчика + байт версии).
        Legacy-формат:      <I I I>   = 12 байт (без версии, рекорды v1).
        Различаем СТРУКТУРНО по длине данных после заголовка:
          - новый: len(payload) == 13 + 4*del_cnt (и далее блоки узлов);
          - legacy: len(payload) == 12 + 4*del_cnt.
        Если оба варианта дают несовпадение — выбираем тот, при котором
        остаток payload допускает разбор блоков (эвристика минимального
        хвоста). Версия из байта 12 используется только как подсказка
        (легальные значения 1/2), потому что в legacy-дельтах там лежат
        данные (первый байт id удаления), а не версия.
        Возвращает (del_cnt, new_cnt, upd_cnt, is_v2, offset).
        """
        if len(payload) < 12:
            raise ValueError("Delta payload too short")
        d, n, u = struct.unpack_from('<I I I', payload, 0)
        del_bytes = 4 * d
        # Кандидат 1: новый 13-байтный заголовок (с байтом версии)
        new_ok = False
        ver_hint = None
        if len(payload) >= 13:
            ver_hint, = struct.unpack_from('<B', payload, 12)
            tail_new = len(payload) - 13 - del_bytes
            new_ok = (tail_new >= 0 and (ver_hint in (1, 2, 3)) and
                      (n == 0 or tail_new > 0))
        # Кандидат 2: legacy 12-байтный заголовок
        tail_old = len(payload) - 12 - del_bytes
        old_ok = (tail_old >= 0 and (n + u == 0 or tail_old > 0))
        if new_ok and not old_ok:
            return d, n, u, (ver_hint or 0) >= 2, 13
        if old_ok and not new_ok:
            return d, n, u, False, 12
        if new_ok and old_ok:
            # Оба разбора структурно допустимы (например, пустая дельта):
            # байт 12 == 1 или 2 → это явная версия нового формата.
            if ver_hint in (1, 2):
                return d, n, u, ver_hint >= 2, 13
            return d, n, u, False, 12
        raise ValueError("Unrecognized delta header layout")

    @staticmethod
    def _load_delta(lattice, buffer: bytes, prev_cache: dict) -> bool:
        stored_crc = struct.unpack('<I', buffer[-4:])[0]
        if zlib.crc32(buffer[:-4]) & 0xFFFFFFFF != stored_crc:
            print("[ПАМЯТЬ] Ошибка CRC32 дельты!")
            return False
        
        orig_size = struct.unpack('<I', buffer[8:12])[0]
        compressed = buffer[12:-4]
        payload = CrystalSnapshot.rle_decompress(compressed)
        
        if len(payload) != orig_size:
            print("[ПАМЯТЬ] Ошибка: Размер RLE не совпадает!")
            return False
        
        # 🚀 ИСПРАВЛЕНО: Читаем три счётчика + байт версии формата с начала payload.
        # Legacy-дельты (до Фазы 0) имели заголовок из 12 байт без версии — для них
        # включаем fallback на v1-формат рекордов. Версия валидируется структурно
        # (см. _delta_header_size), а не по содержимому рекордов, поэтому ложных
        # срабатываний, как у старого probe-эвристического автоопределения, нет.
        del_cnt, new_cnt, upd_cnt, is_v2, offset = CrystalSnapshot._delta_header_size(payload)
        
        # 1. Удаления
        for _ in range(del_cnt):
            pid = struct.unpack_from('<I', payload, offset)[0]
            offset += 4
            if pid in lattice.resonators:
                lbl = lattice.resonators[pid].label
                del lattice.resonators[pid]
                if lbl in lattice.label_to_id: del lattice.label_to_id[lbl]
        
        # 2. Новые (полная распаковка)
        from engine import Resonator, TruthValue, SemanticVector, NodeType
        
        for _ in range(new_cnt):
            id_, lbl_len = struct.unpack_from('<I H', payload, offset)
            if is_v2:
                sem_len, = struct.unpack_from('<H', payload,
                                              offset + 6 + lbl_len + CrystalSnapshot.BYTES_PER_VECTOR)
                fmt = f'<I H {lbl_len}s 1250s H {sem_len}s B I Q'
                (_rid, _ll, lbl_b, hdc_b, _sl, _sb, st, en, tk) = struct.unpack_from(fmt, payload, offset)
                probe_off = offset + 6 + lbl_len + CrystalSnapshot.BYTES_PER_VECTOR
                sem, _ = SemanticVector.from_bytes(payload, probe_off + 2)
            else:
                fmt = f'<I H {lbl_len}s 1250s B I Q'
                id_, _, lbl_b, hdc_b, st, en, tk = struct.unpack_from(fmt, payload, offset)
                sem = SemanticVector()
            offset += struct.calcsize(fmt)
            
            lbl = lbl_b.decode('utf-8')
            r = Resonator(id=id_, label=lbl, hdc_vector=int.from_bytes(hdc_b, 'little'), energy=en, last_tick=tk)
            if lbl.startswith('tok:'):   r.node_type = NodeType.TOKEN
            elif lbl.startswith('lem:'): r.node_type = NodeType.LEMMA
            elif lbl.startswith('op:'):  r.node_type = NodeType.OPERATOR
            elif lbl.startswith('sense:'): r.node_type = NodeType.SENSE
            elif lbl.startswith('ent:'): r.node_type = NodeType.ENTITY
            r.semantic = sem
            r.state = TruthValue(st)
            lattice.resonators[id_] = r
            lattice.label_to_id[lbl] = id_
            
            while True:
                t = struct.unpack_from('<I', payload, offset)[0]
                offset += 4
                if t == 0xFFFFFFFF: break
                w = struct.unpack_from('<I', payload, offset)[0]
                offset += 4
                r.connections[t] = w
            
            ctx_len = struct.unpack_from('<H', payload, offset)[0]
            offset += 2
            for __ in range(ctx_len):
                hdc_b = payload[offset:offset+1250]
                offset += 1250
                r.context_sources.append(int.from_bytes(hdc_b, 'little'))
        
        # 3. Обновления (XOR-декодирование)
        for _ in range(upd_cnt):
            uid, d_st, d_en, d_tk = struct.unpack_from('<I B I Q', payload, offset)
            offset += 17
            if uid in lattice.resonators:
                r = lattice.resonators[uid]
                r.state = TruthValue(int(r.state) ^ d_st)
                r.energy ^= d_en
                r.last_tick ^= d_tk
                r.connections.clear()
                while True:
                    t = struct.unpack_from('<I', payload, offset)[0]
                    offset += 4
                    if t == 0xFFFFFFFF: break
                    w = struct.unpack_from('<I', payload, offset)[0]
                    offset += 4
                    r.connections[t] = w
                r.context_sources.clear()
                ctx_len = struct.unpack_from('<H', payload, offset)[0]
                offset += 2
                for __ in range(ctx_len):
                    hdc_b = payload[offset:offset+1250]
                    offset += 1250
                    r.context_sources.append(int.from_bytes(hdc_b, 'little'))
        
        print(f"[ПАМЯТЬ] ⚡ Эпизод (Дельта) наложен: {len(lattice.resonators)} узлов.")
        return True
    
    @staticmethod
    def test_roundtrip(lattice):
        """🧪 Тест полного цикла сохранения/загрузки дельты."""
        import tempfile
        import os
        
        print("\n🧪 [ТЕСТ] Проверка дельта-снапшотов...")
        
        # Создаём базовый снапшот
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cry') as f:
            base_file = f.name
        CrystalSnapshot.save(lattice, base_file)
        base_cache = CrystalSnapshot.generate_state_cache(lattice)
        
        # Модифицируем состояние
        for r in list(lattice.resonators.values())[:5]:
            r.energy += 100
            r.last_tick = lattice.tick_count
        
        # Создаём дельту
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cdt') as f:
            delta_file = f.name
        CrystalSnapshot.save_delta(lattice, delta_file, base_cache)
        
        # Восстанавливаем базовое состояние
        CrystalSnapshot.load(lattice, base_file)
        original_count = len(lattice.resonators)
        
        # Накладываем дельту
        success = CrystalSnapshot.load(lattice, delta_file, base_cache)
        
        # Проверяем
        if success and len(lattice.resonators) == original_count:
            print("   ✅ Тест пройден: дельта корректно наложена")
        else:
            print(f"   ❌ Тест провален: ожидалось {original_count} узлов, получено {len(lattice.resonators)}")
        
        # Очистка
        os.unlink(base_file)
        os.unlink(delta_file)
        
        return success