"""Multi-relation adjacency with a deterministic single-edge legacy projection.

Energy algorithms intentionally continue using dict.items(). Semantic algorithms
must use typed_items/get_type. No semantic relation can be overwritten by SYN.
"""
class TypedConnections(dict):
    def __init__(self, values=()):
        super().__init__()
        self.relations = {}
        self.update(values)

    def __setitem__(self, target, packed):
        kind = packed & 0xF0000000
        self.relations.setdefault(target, {})[kind] = packed
        # Fixed semantic priority; weight/SYN repetition cannot change edge type.
        choices = self.relations[target]
        priority = {0x10000000:9, 0x20000000:8, 0x60000000:7,
                    0x70000000:6, 0x40000000:5, 0x50000000:4}
        super().__setitem__(target, max(choices.values(), key=lambda p:(priority.get(p & 0xF0000000, 1 if p & 0xF0000000 else 0), p)))

    def get_type(self, target, kind, default=None):
        return self.relations.get(target, {}).get(kind, default)

    def typed_items(self):
        for target, types in self.relations.items():
            for packed in types.values():
                yield target, packed

    def remove_type(self, target, kind):
        types = self.relations.get(target, {})
        types.pop(kind, None)
        if types:
            packed = next(iter(types.values()))
            self[target] = packed
        elif target in self:
            del self[target]

    def __delitem__(self, target):
        super().__delitem__(target)
        self.relations.pop(target, None)

    def pop(self, target, *default):
        if target not in self:
            if default:return default[0]
            raise KeyError(target)
        value=self[target];del self[target];return value

    def clear(self):
        super().clear();self.relations.clear()

    def update(self, values=(), **kwargs):
        values = values.typed_items() if isinstance(values, TypedConnections) else (values.items() if hasattr(values,'items') else values)
        for target, packed in values:self[target]=packed
        for target, packed in kwargs.items():self[target]=packed

    def setdefault(self, target, default=0):
        if target not in self:self[target]=default
        return self[target]
