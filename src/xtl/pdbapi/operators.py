from dataclasses import dataclass
from typing import Union, List

from .options import ComparisonType


TNumber = Union[int, float, date]
TIterable = Union[List[str], List[int], List[float], List[date]]
TValue = Union[str, TNumber]


@dataclass
class _Operator:
    '''
    Base operator class
    '''

    operator: str

    def to_dict(self):
        return {
            'operator': self.operator
        }

    def __invert__(self):
        if hasattr(self, 'negation'):
            self.negation = not self.negation
        return self


@dataclass
class UnstructuredTextOperator(_Operator):
    '''
    Free text search
    '''

    def __init__(self, value: str):
        self.value = value

    def to_dict(self):
        return {
            'value': self.value
        }


@dataclass
class ExactMatchOperator(_Operator):
    '''
    Find exact occurrences of the input value. Case-sensitive.
    '''

    def __init__(self, attribute: str, value: TValue):
        self.attribute = attribute
        self.operator = 'exact_match'
        self.value = value

    def to_dict(self):
        return {
            'attribute': self.attribute,
            'operator': self.operator,
            'value': self.value
        }


@dataclass
class InOperator(_Operator):
    '''
    Find results if any value in a list of input values mathces. Case-sensitive.
    '''

    def __init__(self, attribute: str, value: TIterable, negation=False):
        self.attribute = attribute
        self.negation = negation
        self.operator = 'in'
        self.value = value

    def to_dict(self):
        return {
            'attribute': self.attribute,
            'negation': self.negation,
            'operator': self.operator,
            'value': self.value
        }


@dataclass
class ExistsOperator(_Operator):
    '''
    Check whether a given field contains any value
    '''

    def __init__(self, attribute: str, negation=False):
        self.attribute = attribute
        self.negation = negation
        self.operator = 'exists'

    def to_dict(self):
        return {
            'attribute': self.attribute,
            'negation': self.negation,
            'operator': self.operator
        }

@dataclass
class ContainsWordsOperator(_Operator):
    '''
    Check if field contains any of the given words. Case-insensitive.
    '''

    def __init__(self, attribute: str, value: list[str], negation=False):
        self.attribute = attribute
        self.negation = negation
        self.operator = 'contains_words'
        self.value = value

    def to_dict(self):
        return {
            'attribute': self.attribute,
            'negation': self.negation,
            'operator': self.operator,
            'value': self.value
        }


@dataclass
class ContainsPhraseOperator(_Operator):
    '''
    Check if field contains all of the given words in the same order.
    '''

    def __init__(self, attribute: str, value: str, negation=False):
        self.attribute = attribute
        self.negation = negation
        self.operator = 'contains_phrase'
        self.value = value

    def to_dict(self):
        return {
            'attribute': self.attribute,
            'negation': self.negation,
            'operator': self.operator,
            'value': self.value
        }


@dataclass
class ComparisonOperator(_Operator):
    '''
    Perform mathematical comparisons on the field's value.
    '''

    def __init__(self, attribute: str, value: TNumber, operation: ComparisonType, negation=False):
        self.attribute = attribute
        self.negation = negation
        self.operator = operation.value
        self.value = value

    def to_dict(self):
        return {
            'attribute': self.attribute,
            'negation': self.negation,
            'operator': self.operator,
            'value': self.value
        }


@dataclass
class RangeOperator(_Operator):
    '''
    Match values within a provided range
    '''

    def __init__(self, attribute: str, value_from: TNumber, value_to: TNumber, inclusive_lower=False,
                 inclusive_upper=False, negation=False):
        self.attribute = attribute
        self.negation = negation
        self.operator = 'range'
        self.value = [value_from, value_to]
        self.inclusive = [inclusive_lower, inclusive_upper]

    def to_dict(self):
        return {
            'attribute': self.attribute,
            'negation': self.negation,
            'operator': self.operator,
            'value': {
                'from': self.value[0],
                'include_lower': self.inclusive[0],
                'to': self.value[1],
                'include_upper': self.inclusive[1]
            }
        }


OPERATORS_TEXT = [
    ExactMatchOperator,
    ExistsOperator,
    InOperator,
    ContainsWordsOperator,
    ContainsPhraseOperator,
    ComparisonOperator,
    RangeOperator
]