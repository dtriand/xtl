from dataclasses import dataclass
from typing import overload, Union, List
from datetime import date

from .nodes import QueryField
from .operators import *
from .options import ComparisonType

TNumber = Union[int, float, date]
TIterable = Union[List[str], List[int], List[float], List[date]]
TValue = Union[str, TNumber]

Number = (int, float, date)


@dataclass
class Attribute:

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

    def exact_match(self, value: str):
        return QueryField(ExactMatchOperator(attribute=self.name, value=value))

    def exists(self):
        return QueryField(ExistsOperator(attribute=self.name))

    def in_(self, value: Union[str, TNumber]):
        return QueryField(InOperator(attribute=self.name, value=value))

    def contains_word(self, value: List[str]):
        return QueryField(ContainsWordsOperator(attribute=self.name, value=value))

    def contains_phrase(self, value: str):
        return QueryField(ContainsPhraseOperator(attribute=self.name, value=value))

    def equals(self, value: TNumber):
        return QueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.EQUAL, value=value))

    def greater(self, value: TNumber):
        return QueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.GREATER, value=value))

    def greater_or_equal(self, value: TNumber):
        return QueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.GREATER_OR_EQUAL,
                                             value=value))

    def less(self, value: TNumber):
        return QueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.LESS, value=value))

    def less_or_equal(self, value: TNumber):
        return QueryField(ComparisonOperator(attribute=self.name, operation=ComparisonType.LESS_OR_EQUAL, value=value))

    def range(self, value_from: TNumber, value_to: TNumber, inclusive_lower=False, inclusive_upper=False):
        return QueryField(RangeOperator(attribute=self.name, value_from=value_from, value_to=value_to,
                                        inclusive_lower=inclusive_lower, inclusive_upper=inclusive_upper))

    @overload
    def __eq__(self, other: 'Attribute') -> bool: ...

    @overload
    def __eq__(self, other: str) -> QueryField: ...

    @overload
    def __eq__(self, other: TNumber) -> QueryField: ...

    def __eq__(self, other: Union['Attribute', str, TNumber]) -> Union[QueryField, bool]:
        if isinstance(other, Attribute):
            return self.name == other.name
        elif isinstance(other, str):
            return self.exact_match(other)
        elif isinstance(other, Number):
            return self.equals(other)
        else:
            raise TypeError("other must be one of: 'Attribute', 'str', 'int', 'float' or 'date'")

    @overload
    def __ne__(self, other: 'Attribute') -> bool: ...

    @overload
    def __ne__(self, other: str) -> QueryField: ...

    @overload
    def __ne__(self, other: TNumber) -> QueryField: ...

    def __ne__(self, other: Union['Attribute', str, TNumber]) -> Union[QueryField, bool]:
        if isinstance(other, Attribute):
            return self.name != other.name
        elif isinstance(other, str):
            return ~(self.exact_match(other))
        elif isinstance(other, (int, float, date)):
            return ~(self.equals(other))
        else:
            raise TypeError("other must be one of: 'Attribute', 'str', 'int', 'float' or 'date'")

    def __lt__(self, other: TNumber) -> QueryField:
        if isinstance(other, Number):
            return self.less(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    def __le__(self, other: TNumber) -> QueryField:
        if isinstance(other, Number):
            return self.less_or_equal(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    def __gt__(self, other: TNumber) -> QueryField:
        if isinstance(other, Number):
            return self.greater(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    def __ge__(self, other: TNumber) -> QueryField:
        if isinstance(other, Number):
            return self.greater_or_equal(other)
        else:
            raise TypeError("other must be one of: 'int', 'float' or 'date'")

    def __contains__(self, item: Union[str, List[str]]) -> QueryField:
        if isinstance(item, str):
            return self.contains_phrase(item)
        elif isinstance(item, list) and isinstance(item[0], str):
            return self.contains_word(item)
        else:
            raise NotImplementedError


class AttributeGroup:

    @property
    def children(self):
        return [str(c) for c in self.__dict__]
