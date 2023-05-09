import os

import tqdm


def init_folder(file_path):
    folder = os.path.split(file_path)[0]
    if len(folder) == 0:
        folder = "./"
    os.makedirs(folder, exist_ok=True)


class CorpusSearchIndex:
    def __init__(self, file_path, encoding="utf8", cache_freq=1, sampling=None):
        init_folder(file_path)
        self.fpath = file_path
        self._encoding = encoding
        self._cache_freq = cache_freq
        self._sampling = sampling
        self._build_index()

    def _build_index(self):
        self._lookup, self._numrow, self._doc2idx = [0], 0, {}
        with open(self.fpath, "a+", encoding=self._encoding) as f:
            f.seek(0)
            while self._numrow != self._sampling:
                row = f.readline()
                if len(row) == 0:
                    break
                self._doc2idx[row] = 0
                self._numrow += 1
                if self._numrow % self._cache_freq == 0:
                    self._lookup.append(f.tell())

    def __contains__(self, element):
        return element in self._doc2idx

    def __iter__(self):
        with open(self.fpath, encoding=self._encoding) as f:
            for idx, row in enumerate(f, 1):
                yield row.strip()
                if idx == self._numrow:
                    break

    def __len__(self):
        return self._numrow

    def __getitem__(self, index):
        with open(self.fpath, encoding=self._encoding) as f:
            cacheid = index // self._cache_freq
            f.seek(self._lookup[cacheid])
            for idx, row in enumerate(f, cacheid * self._cache_freq):
                if idx == index:
                    return row.strip()
        raise IndexError("Index %d is out of boundary" % index)

    def append(self, document):
        with open(self.fpath, "a+", encoding=self._encoding) as f:
            f.write(document.replace("\n", "") + "\n")
            self._numrow += 1
            self._doc2idx[document] = len(self._doc2idx)
            if self._numrow % self._cache_freq == 0:
                self._lookup.append(f.tell())

    def clear(self):
        os.remove(self.fpath)
        self._build_index()


class PublicCorpus:
    def __init__(self, corpus, subset, key, start=0, stop=None, skip=1):
        from datasets import load_dataset
        self.corpus = load_dataset(*corpus)[subset]
        if stop is None or stop > len(self.corpus):
            stop = len(self.corpus)
        assert isinstance(start, int) and isinstance(stop, int)
        self.start, self.stop, self.skip = start, stop, skip
        self.key = key

    def __len__(self):
        return len(self.corpus)

    def __iter__(self):
        for idx in tqdm.tqdm(range(self.start, self.stop, self.skip)):
            yield self.corpus[idx][self.key]

    def localize(self, file_path, min_words=3, encoding="utf8", sep="\n"):
        init_folder(file_path)
        with open(file_path, "w", encoding=encoding) as f:
            for doc in self:
                group, words = [], 0.0
                paras = doc.split(sep) if sep else [doc]
                for para in paras:
                    group.append(para.strip().replace("\n", ""))
                    words += group[-1].count(" ") + 1

                    if words >= min_words:
                        f.write(" ".join(group) + "\n")
                        group.clear()
                        words = 0.0

                if words >= 2:
                    f.write(" ".join(group) + "\n")
