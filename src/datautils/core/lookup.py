class LookupTable:
    def __init__(self, maximum=None):
        self.str2id, self.id2str = {}, []
        self.num_limit = maximum

    def __len__(self):
        return len(self.str2id)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.id2str[idx]
        return self.str2id[idx]

    def add(self, item):
        if item not in self.str2id:
            assert len(self.str2id) != self.num_limit
            self.str2id[item] = len(self.str2id)
            self.id2str.append(item)

    def save(self, fpath):
        with open(fpath, "w") as f:
            pickle.dump({"str2id": self.str2id,
                         "id2str": self.id2str},
                         f, protocol=3)

    @classmethod
    def load(cls, fpath):
        table = cls()
        with open(fpath) as f:
            data = pickle.load(f)
        table.str2id = data["str2id"]
        table.id2str = data["id2str"]
        return table

    @classmethod
    def from_txt(cls, fpath, encoding="utf8"):
        with open(fpath, encoding=encoding) as f:
            return cls.from_iter(map(lambda x: x.strip(), f))

    @classmethod
    def from_iter(cls, iterable):
        table = cls()
        for each in iterable:
            table.add(each)
        return table
