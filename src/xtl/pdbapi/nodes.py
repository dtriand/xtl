from dataclasses import dataclass

from .options import NodeType, SearchService, LogicalOperator
from .operators import _Operator


@dataclass
class SearchQueryNode:
    '''
    Base class for a query node
    '''

    def __init__(self, type_: NodeType):
        self.type = type_


@dataclass
class SearchQueryField(SearchQueryNode):
    '''
    A single field query
    '''

    def __init__(self, parameters, service=SearchService.TEXT):
        if not issubclass(parameters.__class__, _Operator):
            raise
        super().__init__(type_=NodeType.TERMINAL)
        self.parameters = parameters
        if service not in (SearchService.TEXT, SearchService.FULL_TEXT):
            raise NotImplementedError
        self.service = service

    def to_dict(self):
        return {
            'type': self.type.value,
            'service': self.service.value,
            'parameters': self.parameters.to_dict()
        }

    def __and__(self, other):
        return SearchQueryGroup(nodes=[self, other], logical_operator=LogicalOperator.AND)

    def __or__(self, other):
        return SearchQueryGroup(nodes=[self, other], logical_operator=LogicalOperator.OR)

    def __invert__(self):
        if hasattr(self.parameters, 'negation'):
            self.parameters.negation = not self.parameters.negation
        return self


@dataclass
class SearchQueryGroup(SearchQueryNode):
    '''
    A multi-field query
    '''

    def __init__(self, nodes: list, logical_operator=LogicalOperator.AND):
        for node in nodes:
            if not issubclass(node.__class__, SearchQueryNode):
                raise
        super().__init__(type_=NodeType.GROUP)
        self.logical_operator = logical_operator
        self.nodes = nodes

    def to_dict(self):
        return {
            'type': self.type.value,
            'logical_operator': self.logical_operator.value,
            'nodes': [node.to_dict() for node in self.nodes]
        }

    def __and__(self, other):
        if isinstance(other, SearchQueryField):
            self.nodes.append(other)
            return self
        elif isinstance(other, SearchQueryGroup):
            return SearchQueryGroup(nodes=[self, other], logical_operator=LogicalOperator.AND)

    def __or__(self, other):
        return SearchQueryGroup(nodes=[self, other], logical_operator=LogicalOperator.OR)


