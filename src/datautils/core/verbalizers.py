import bisect
import math

import zipcodes

class Simple:
    def __init__(self, variable, missing=""):
        assert isinstance(variable, str)
        assert isinstance(missing, str)
        self.name = variable
        self.miss = missing
        self._val = {}

    def _render(self, value):
        return value

    def verbalize(self, value):
        if self.miss == value:
            return None
        return self._render(value)

    def vectorize(self, value):
        value = self.verbalize(value)
        if value is None:
            return 0
        if isinstance(value, float):
            return value
        return self._val.setdefault(value, len(self._val) + 1)


class Continue(Simple):
    def __init__(self, variable, missing, splits, groups):
        Simple.__init__(self, variable, missing)
        assert isinstance(groups, list)
        assert isinstance(splits, list)
        assert len(groups) == len(splits) + 1
        self._var = groups
        self._cut = splits

    def _render(self, value):
        return self._var[bisect.bisect_left(self._cut, float(value))]

    def vectorize(self, value):
        if value == self.miss:
            return 0.0
        return float(value)


class Category(Simple):
    def __init__(self, variable, missing, lookup_table):
        assert isinstance(lookup_table, dict)
        Simple.__init__(self, variable, missing)
        self.lookup = lookup_table

    def _render(self, value):
        return self.lookup.get(value)


class Nominal(Simple):
    def __init__(self, variable, mapper, split="|", combine=(", ", "and"), missing=""):
        assert isinstance(mapper, dict) or callable(mapper)
        assert isinstance(combine, tuple) and len(combine) == 2
        Simple.__init__(self, variable, missing)
        self.mapper = mapper
        self.seg = split
        self.precomb, self.lastcomb = combine

    def _render(self, value):
        values = value.split(self.seg)
        if callable(self.mapper):
            values = [self.mapper(_) for _ in values if _ != self.miss]
        else:
            values = [self.mapper[_] for _ in values if _ in self.lookup]
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]
        return self.precomb.join(values[:-1]) + self.precomb + self.lastcomb + values[-1]


class Functional(Simple):
    def __init__(self, variable, function, missing=""):
        assert callable(function)
        Simple.__init__(self, variable, missing)
        self._func = function

    def _render(self, value):
        return self._func(value)


class Zipcode(Simple):
    def __init__(self, name, miss):
        Simple.__init__(self, name, miss)
        self._states = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
                        "CA": "California", "CO": "Colorado", "CT": "Connecticut",
                        "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
                        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
                        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
                        "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "mississippi", "MO": "Missouri", "MT": "Montana",
                        "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
                        "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
                        "PA": "Pennsylvania", "RI": "Rhode Island",
                        "SC": "South Carolina", "SD": "South Dakota",
                        "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
                        "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}

    def _render(self, value):
        try:
            return self._states[zipcodes.matching(value)[0]["state"]]
        except Exception:
            return None


class Occupation(Simple):
    def __init__(self, name):
        Simple.__init__(self, name, "?")

    def _render(self, value):
        if value in ("other", "none"):
            return "freelance"
        if value == "markting":
            return "seller"
        return value

class MovieTitle(Simple):
    def __init__(self, name):
        Simple.__init__(self, name, "N/A")

    def _render(self, title):
        if title == "unknown":
            return title
        title, year = title[:-7], title[-5:-1]
        if "," in title:
            left, right = title.rsplit(",", 1)
            title = right + " " + left
        return title
