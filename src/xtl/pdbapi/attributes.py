from dataclasses import dataclass, field
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

    fullname: str
    type: str
    description: str = field(repr=False)
    contains: list[str] = ''  # nested datatype for arrays
    name: str = ''
    parent: str = ''

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
        return SearchQueryField(ExactMatchOperator(attribute=self.fullname, value=value))

    def exists(self):
        return SearchQueryField(ExistsOperator(attribute=self.fullname))

    def in_(self, value: Union[str, TNumber]):
        return SearchQueryField(InOperator(attribute=self.fullname, value=value))

    def contains_word(self, value: List[str]):
        return SearchQueryField(ContainsWordsOperator(attribute=self.fullname, value=value))

    def contains_phrase(self, value: str):
        return SearchQueryField(ContainsPhraseOperator(attribute=self.fullname, value=value))

    def equals(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.fullname, operation=ComparisonType.EQUAL,
                                                   value=value))

    def greater(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.fullname, operation=ComparisonType.GREATER,
                                                   value=value))

    def greater_or_equal(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.fullname, operation=ComparisonType.GREATER_OR_EQUAL,
                                                   value=value))

    def less(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.fullname, operation=ComparisonType.LESS, value=value))

    def less_or_equal(self, value: TNumber):
        return SearchQueryField(ComparisonOperator(attribute=self.fullname, operation=ComparisonType.LESS_OR_EQUAL,
                                                   value=value))

    def range(self, value_from: TNumber, value_to: TNumber, inclusive_lower=False, inclusive_upper=False):
        return SearchQueryField(RangeOperator(attribute=self.fullname, value_from=value_from, value_to=value_to,
                                              inclusive_lower=inclusive_lower, inclusive_upper=inclusive_upper))

    @overload
    def __eq__(self, other: 'SearchAttribute') -> bool: ...

    @overload
    def __eq__(self, other: str) -> SearchQueryField: ...

    @overload
    def __eq__(self, other: TNumber) -> SearchQueryField: ...

    def __eq__(self, other: Union['SearchAttribute', str, TNumber]) -> Union[SearchQueryField, bool]:
        if isinstance(other, SearchAttribute):
            return self.fullname == other.fullname
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
            return self.fullname != other.fullname
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


class DataAttribute(_Attribute):

    def __eq__(self, other: 'DataAttribute') -> bool:
        if isinstance(other, DataAttribute):
            return self.fullname == other.fullname
        else:
            raise TypeError("other must be of type 'DataAttribute'")


def _make_empty_list():
    return []


@dataclass
class _AttributeGroup:
    name_: str
    parent_name: str = ''
    _children: list[str] = field(init=False, default_factory=_make_empty_list)
    parent: '_AttributeGroup' = None

    def update_children(self):
        self._children = [str(c) for c in self.__dict__]
        self._children.remove('name_')
        self._children.remove('parent_name')
        self._children.remove('_children')
        self._children.remove('parent')
        try:
            self._children.remove('_schema')
        except ValueError:
            pass

    @property
    def children(self):
        self.update_children()
        return self._children

    @property
    def attributes(self) -> list[_Attribute]:
        return [getattr(self, a) for a in self.children]


class SearchAttributeGroup(_AttributeGroup):

    ...


class DataAttributeGroup(_AttributeGroup):

    ...
