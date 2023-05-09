import re
import multiprocessing

import transformers


class Slot:
    
    __slots__ = ["name", "pos", "maxlen", "optional", "prefix", "suffix", "tokenize", "fid"]
    __autofid__ = 5
    
    def __init__(self, name, pos, tokenizer, maxlen=None, optional=False, prefix=None, suffix=None, fid=None):
        if fid is None:
            fid = Slot.__autofid__ + 1
            Slot.__autofid__ += 1
        self.name = name
        self.pos = pos
        self.fid = fid
        self.maxlen = maxlen
        self.optional = optional
        self.prefix = prefix
        self.suffix = suffix
        self.tokenize = tokenizer.tokenize

    def __repr__(self):
        return "Slot(%s, fid=%s, at=%s)" % (self.name, self.fid, self.pos)

    def render(self, tokens):
        return self.tokenize(self.prefix + tokens + self.suffix)[:self.maxlen]

        
class Template:    
    def __init__(self, prompt, tokenizer, maxlen, slot_args=None):
        assert isinstance(prompt, str), "`prompt` keyword is a string."
        assert "{:" in prompt and ":}" in prompt, "Template contains at least one slot like `{:NAME:}`"
        self.configs = slot_args if slot_args else {}
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.sep_token = tokenizer.sep_token if hasattr(tokenizer, "sep_token") else tokenizer.eos_token
        if "[::MASK::]" in prompt:
            if self.tokenizer.mask_token is None:
                prompt = prompt[:prompt.index("[::MASK::]")]
            else:
                prompt = prompt.replace("[::MASK::]", self.tokenizer.mask_token)
        self.blank = self._init_blank(prompt)
        self.slots, self.prompt = self._parse_prompt(prompt)
        self.maxlen = maxlen
        self.sep_token = tokenizer.sep_token if hasattr(tokenizer, "sep_token") else tokenizer.eos_token

    def _parse_prompt(self, prompt):            
        slots = []
        chars = list(prompt)
        for sid, slot in enumerate(re.finditer(r"(\{\:.+?\:\})", prompt)):
            head, tail = slot.span()
            chars[head:tail] = [self.blank, " "] + [""] * (tail - head - 2)
            name = slot.group()[2:-2]
            slot = self.configs.get(name, {})
            if not isinstance(slot, Slot):
                slot = Slot(
                        name=name, pos=None,
                        tokenizer=self.tokenizer,
                        maxlen=slot.get("maxlen", None),
                        optional=slot.get("optional", False),
                        prefix=slot.get("prefix", ""),
                        suffix=slot.get("suffix", ""),
                        fid=slot.get("fid", None)
                        )
            slots.append(slot)
                        
        tmp = iter(slots)
        sentence = re.sub(r"(\ ){2,}", " ", u"".join(chars))
        tokens = [_ for _ in self.tokenizer.tokenize(sentence) if len(_.strip()) > 0]
        blanks = {self.blank, self.tokenizer.tokenize(" %s" % self.blank)[-1]}
        for pos, token in enumerate(tokens):
            if token in blanks:
                next(tmp).pos = pos
                
        for slot in slots:
            assert slot.pos is not None, "Template construction is faild at slot: %s" % slot
        return slots, tokens

    def _init_blank(self, prompt, word=None):
        if word is not None:
            assert word in self.tokenizer.vocab
            return word
        for word in sorted(self.tokenizer.vocab):
            if not word.isdigit() and\
               word not in prompt and\
               len(word) > 2 and \
               word != self.sep_token and\
               word[0] in {"[", "<"} and\
               word[-1] in {"]", ">"}:
                return word
        raise RuntimeError("Cannot find a free blank word.")

    def render(self, **kwrds):
        feature, sample, shift = [0] * len(self.prompt), self.prompt.copy(), 0
        for slot in self.slots:
            position = shift + slot.pos
            if slot.name not in kwrds or kwrds[slot.name] is None:
                assert slot.optional, "missing required slot: %s" % slot.name
                del sample[position], feature[position]
                shift -= 1
                continue
            fill = slot.render(kwrds[slot.name])
            sample[position: position+1] = fill
            feature[position: position+1] = [slot.fid] * len(fill)
            shift += len(fill) - 1
        return feature, sample 

    def construct(self, **kwrds):
        features, tokens = self.render(**kwrds)
        size = len(tokens)
        assert size <= self.maxlen, "text doesn't fit `max_len` requirement: %d words" % self.maxlen
        padding = [0] * (self.maxlen - size)
        segs = []
        for idx, token in enumerate(tokens):
            if token == self.sep_token:
                break
        segs = [0] * idx + [1] * (self.maxlen - idx)
        ids = self.tokenizer.convert_tokens_to_ids(tokens) + padding
        mask = [1] * size + padding
        features = features + padding
        return " ".join(tokens), ids, mask, features, segs

    def render_batch(self, batch):
        return [self.render(**_) for _ in batch]

    def get_target_id(self, target_name):
        assert isinstance(target_name, str)
        for slot in self.slots:
            if slot.name == target_name:
                return slot.fid
        raise RuntimeError("``%s`` is not a valid slot name to the template: %s" % (target_name, self))



if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    prompt = "a customer is a {:age:} {:gender:} {:location:}. " +\
             "The product is a {:name:} produced by {:brand:}. " +\
             "{:weather:}" +\
             "Question: Whether the customer will buy the product? Answer: [MASK]."
    t = Template(prompt, tokenizer, 512,
                 {"weather": {"prefix": "Today is a", "suffix": "day.", "optional": True},
                  "location": {"prefix": "living at", "optional": True}})
    print(t.construct(location="Nanning", age="young", gender="woman", name="basketball shoes", brand="Nike", weather="sunny"))
    print(t.render_batch([dict(age="young", gender="woman", name="basketball shoes", brand="Nike", weather="sunny")] * 4))
