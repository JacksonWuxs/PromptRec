import math
import re
import uuid
import collections
import multiprocessing

import nltk
import tqdm

from .corpus import CorpusSearchIndex



def jaccard(left, right, levels):
    return nltk.distance.jaccard_distance(
            set(nltk.ngrams(left, levels)),
            set(nltk.ngrams(right, levels)),
            )


class InvertedIndex:
    def __init__(self, k1=1.2, k2=1.2, b=0.75, max_words=None, ngrams=1):
        self._wordlist = {}
        self._maxwords = max_words
        self._docs = 0
        self._k1 = k1
        self._k2 = k2
        self._b = b

    def __len__(self):
        return len(self._wordlist)

    def _parallel_insert(self, inputs, outputs, tokenize=None):
        while True:
            if not inputs.empty():
                item = inputs.get(True)
                try:
                    if item is None:
                        inputs.put(None)
                        outputs.put(self._wordlist)
                        break
                    idx, text = item
                    if callable(tokenize):
                        text = tokenize(text)
                    self.insert(idx, text)
                except:
                    inputs.put(item)

    def _squeeze(self, size=None):
        while len(self._wordlist) > 0:
            word_freq = sorted((len(_[1]), _[0]) \
                               for _ in self._wordlist.items())
            drop_freq = word_freq[0][0]
            for freq, word in word_freq:
                if freq != drop_freq:
                    break
                del self._wordlist[word]
            if size is None or len(self._wordlist) <= min(self._maxwords, size):
                break

    def insert(self, idx, tokens):
        for token in set(tokens):
            if token not in self._wordlist:
                if len(self._wordlist) == self._maxwords:
                    self._squeeze()
                self._wordlist[token] = []
            self._wordlist[token].append(idx)
        self._docs += 1

    def retrieve(self, tokens):
        candidates = set()
        for token in tokens:
            cands = set(self._wordlist.get(token, []))
            if len(candidates) == 0:
                candidates = cands
            elif len(cands) > 0:
                candidates = candidates & cands
        return sorted(candidates)

    def clear(self):
        self._wordlist.clear()

    def build(self, documents, tokenize=None, num_workers=None, queues=None):
        inputs, outputs = queues
        maxwords, self._maxwords = self._maxwords, None
        try:
            for worker in range(num_workers):
                multiprocessing.Process(target=self._parallel_insert,
                                        args=(inputs, outputs, tokenize)
                                        ).start()
            
            for idx, doc in documents:
                assert isinstance(idx, int) and isinstance(doc, str) 
                inputs.put((idx, doc))
                self._docs += 1
        finally:
            inputs.put(None)
            self._maxwords = maxwords

        for idx in range(num_workers):
            for token, files in outputs.get(True).items():
                self._wordlist.setdefault(token, []).extend(files)
        assert inputs.get(True) is None
        self._squeeze(self._maxwords)

    def ranking(self, qry, docIDs, docTokens):
        qry_idfs = {w: math.log10(self._docs / len(self._wordlist.get(w, [None]))) for w in qry}
        qry_counts, doc_counts = collections.Counter(qry), []
        for docID, tokens in zip(docIDs, docTokens):
            doc_counts.append(collections.Counter(tokens))
            
        doc_scores = []
        Lavg = sum(sum(count.values()) for count in doc_counts) / len(doc_counts)
        k1, k2, b = self._k1, self._k2, self._b
        for docID, doc_count in zip(docIDs, doc_counts):
            docS = 0.0
            Ld = sum(doc_count.values())
            K = 1.0 - b + b * Ld / Lavg 
            for token, tftq in qry_counts.items():
                tftd = doc_count[token]# / Ld 
                s1 = qry_idfs[token]
                s2 = (k1 + 1) * tftd / (k1 * K + tftd)
                s3 = (k2 + 1) * tftq / (k2 + tftq)
                docS += s1 * s2 * s3
            doc_scores.append((docS, docID))
        return [_[1] for _ in sorted(doc_scores, reverse=True)]


class Tokenizer:
    def __init__(self):
        self._tokenize = nltk.word_tokenize
        self._stemmer = nltk.stem.PorterStemmer()
        self._stopwords = set(nltk.corpus.stopwords.words("english")) | {"n't", "asap", "lol", "luv", "wtg"}
        self._cleaner = re.compile(r"(@\[a-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?")

    def tokenize(self, text):
        tokens = []
        text = self._cleaner.sub("", text.lower())
        for token in self._tokenize(text):
            if token.isalpha() and len(set(token)) >= 2 and token not in self._stopwords:
                tokens.append(self._stemmer.stem(token))
        return tokens


class SearchEngineSystem:
    def __init__(self, cache_path=None, sampling=None, cache_freq=1, ngrams=2, max_words=None, workers=None):
        if cache_path is None:
            cache_path = str(uuid.uuid1) + ".txt"
        if workers is None:
            workers = multiprocessing.cpu_count() - 1
        self._workers = workers
        self._inputs = multiprocessing.Queue(workers)
        self._outputs = multiprocessing.Queue()
        self._running = False
        self._documents = CorpusSearchIndex(cache_path, cache_freq=cache_freq, sampling=sampling)
        self._inverted_index = InvertedIndex(10.0, 1.2, 1.0, max_words, ngrams)
        self._tokenizer = Tokenizer()
        
        self._tokenize = self._tokenizer.tokenize
        self._insert = self._inverted_index.insert
        self._retrieve = self._inverted_index.retrieve
        self._ranking = self._inverted_index.ranking
        self._append = self._documents.append

        if len(self._documents) > 0:
            documents = tqdm.tqdm(enumerate(self._documents), total=len(self._documents), desc="BuildEngine")
            self._inverted_index.build(documents, self._tokenize, self._workers, (self._inputs, self._outputs))

    def _parallel_search(self, inputs, outputs):
        while True:
            if not inputs.empty():
                item = inputs.get(True)
                if item is None:
                    inputs.put(None)
                    outputs.put(None)
                    break
                outputs.put((item[0], self.search(item[1], item[2])))

    def insert(self, new_docs):
        if not isinstance(new_docs, str):
            return sum(self.insert(_) for _ in new_docs)
        new_doc = new_docs.strip()
        if new_doc not in self._documents:
            tokens = self._tokenize(new_doc)
            self._insert(len(self._documents), tokens)
            self._append(new_doc)
            return len(tokens)
        return 0

    def search(self, query, topK=None):
        query = self._tokenize(query)
        docIDs = self._retrieve(query)
        if topK is not None and len(docIDs) > topK:
            docs = (self._tokenize(self._documents[did]) for did in docIDs)
            docIDs = self._ranking(query, docIDs, docs)
        return [self._documents[did] for did in docIDs[:topK]]
    
    def batch_search(self, queries, topK=None):
        assert self._inputs.empty() and self._outputs.empty(), "Engine is still busy."
        for idx, query in enumerate(queries):
            self._inputs.put((idx, query, topK))
        documents = [None] * (idx + 1)
        for _ in range(idx + 1):
            idx, rslt = self._outputs.get(True)
            documents[idx] = rslt
        return documents

    def parallize(self, workers=None):
        if workers is None:
            workers = self._workers
        assert self._running is False
        for worker in range(workers):
            multiprocessing.Process(target=self._parallel_search,
                                    args=(self._inputs, self._outputs)
                                    ).start()
        self._running = True

    def terminate(self):
        assert self._inputs.empty() and self._outputs.empty(), "Engine is still busy."
        self._inputs.put(None)
        for worker in range(self._workers):
            assert self._outputs.get(True) is None
        self._running = False


    def summary(self):
        import time
        print("\nSearching Engine Self-Report")
        print("Time: %s" % time.asctime())
        print("File Path: %s" % self._documents.fpath)
        print("Number of Documents: %d" % len(self._documents))
        print("Number of Tokens: %d" % len(self._inverted_index))
        print("Maximize Workers: %d" % self._workers)

    def clear(self):
        self._inverted_index.clear()
        self._documents.clear()


if __name__ == "__main__":
    import time
    engine = SearchEngineSystem("./reuters_searching.txt", workers=4, sampling=100)
    engine.summary()
    engine.clear()
    spent = 0.0
    reuters = nltk.corpus.reuters
    for fileid in reuters.fileids():
        news = " ".join(reuters.words(fileid))
        begin = time.time()
        engine.insert(news)
        spent += time.time() - begin
    print("Spent %2f seconds to build index." % spent)
    engine.summary()
    queries = ["China Japan"] * 1000

    engine.parallize()
    begin = time.time()
    engine.batch_search(queries)
    print("Batch Spent: %.4f" % (time.time() - begin,))
    engine.terminate()

    begin = time.time()
    rslts = []
    for query in tqdm.tqdm(queries):
        rslts.append(engine.search(query))
    print("Single Spent: %.4f" % (time.time() - begin,))

