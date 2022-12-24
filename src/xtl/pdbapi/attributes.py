from typing import overload

from xtl.pdbapi.search.nodes import SearchQueryField
from xtl.pdbapi.search.operators import *
from xtl.pdbapi.search.options import ComparisonType

TNumber = Union[int, float, date]
TIterable = Union[list[str], list[int], list[float], list[date]]
TValue = Union[str, TNumber]

Number = (int, float, date)


@dataclass
class _Attribute:

    name: str
    type: str
    description: str

    def _type_checking(self, value):
        if self.type == 'integer':
            if not isinstance(value, int):
                raise TypeError
        elif self.type == 'string':
            if not isinstance(value, str):
                raise TypeError
        elif self.type == 'number':
            if not isinstance(value, Number):
                raise TypeError


@dataclass
class SearchAttribute(_Attribute):

    def exact_match(self, value: str):
        return SearchQueryField(ExactMatchOperator(attribute=self.name, value=value))

    def exists(self):
        return SearchQueryField(ExistsOperator(attribute=self.name))

    def in_(self, value: Union[str, TNumber]):
        return SearchQueryField(InOperator(attribute=self.name, value=value))

    def contains_word(self, value: List[str]):
        return SearchQueryField(ContainsWordsOperator(attribute=self.name, value=value))

    def contains_phrase(self, value: str):
        return SearchQueryField(ContainsPhraseOperator(attribute=self.name, value=value))

    def equals(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.EQUAL, value=value))

    def greater(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.GREATER, value=value))

    def greater_or_equal(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.GREATER_OR_EQUAL,
                                                   value=value))

    def less(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.LESS, value=value))

    def less_or_equal(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.LESS_OR_EQUAL, value=value))

    def range(self, value_from: TNumber, value_to: TNumber, inclusive_lower=False, inclusive_upper=False):
        return SearchQueryField(RangeOperator(attribute=self.name, value_from=value_from, value_to=value_to,
                                              inclusive_lower=inclusive_lower, inclusive_upper=inclusive_upper))

    @overload
    def __eq__(self, other: 'SearchAttribute') -> bool: ...

    @overload
    def __eq__(self, other: str) -> SearchQueryField: ...

    @overload
    def __eq__(self, other: TNumber) -> SearchQueryField: ...

    def __eq__(self, other: Union['SearchAttribute', str, TNumber]) -> Union[SearchQueryField, bool]:
        if isinstance(other, SearchAttribute):
            return self.name == other.name
        elif isinstance(other, str):
            return self.exact_match(other)
        elif isinstance(other, Number):
            return self.equals(other)
        else:
            raise TypeError("other must be one of: 'SearchAttribute', 'str', 'int', 'float' or 'date'")

    @overload
    def __ne__(self, other: 'SearchAttribute') -> bool: ...

    @overload
    def __ne__(self, other: str) -> SearchQueryField: ...

    @overload
    def __ne__(self, other: TNumber) -> SearchQueryField: ...

    def __ne__(self, other: Union['SearchAttribute', str, TNumber]) -> Union[SearchQueryField, bool]:
        if isinstance(other, SearchAttribute):
            return self.name != other.name
        elif isinstance(other, str):
            return ~(self.exact_match(other))
        elif isinstance(other, (int, float, date)):
            return ~(self.equals(other))
        else:
            raise TypeError("other must be one of: 'SearchAttribute', 'str', 'int', 'float' or 'date'")

    def __lt__(self, other: TNumber) -> SearchQueryField:
        if isinstance(other, Number):
            return self.less(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    def __le__(self, other: TNumber) -> SearchQueryField:
        if isinstance(other, Number):
            return self.less_or_equal(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    def __gt__(self, other: TNumber) -> SearchQueryField:
        if isinstance(other, Number):
            return self.greater(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    def __ge__(self, other: TNumber) -> SearchQueryField:
        if isinstance(other, Number):
            return self.greater_or_equal(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    # def __contains__(self, item: Union[str, list[str]]) -> QueryField:
    #     if isinstance(item, str):  # attr in 'xxx'
    #         return self.contains_phrase(item)
    #     elif isinstance(item, list) and isinstance(item[0], str):  # attr in ['xxx', 'yyy']
    #         return self.contains_word(item)
    #     else:
    #         raise NotImplementedError


class SearchAttributeGroup:

    @property
    def children(self):
        return [str(c) for c in self.__dict__]
