from dataclasses import dataclass
from typing import overload

from xtl.pdbapi.attributes import DataAttribute
from xtl.exceptions import InvalidArgument


@dataclass
class DataQueryField:

    def __init__(self, attribute: DataAttribute):
        if not isinstance(attribute, DataAttribute):
            raise InvalidArgument(raiser='attribute', message='Must be of type \'DataAttribute\'')
        self.attribute = attribute

    def to_gql(self):
        if not self.attribute.parent:
            return self.attribute.name
        return f'{self.attribute.parent}{{{self.attribute.name}}}'

    @overload
    def __add__(self, other: DataAttribute) -> 'DataQueryGroup': ...

    @overload
    def __add__(self, other: 'DataQueryField') -> 'DataQueryGroup': ...

    def __add__(self, other: DataAttribute or 'DataQueryField') -> 'DataQueryGroup':
        if isinstance(other, DataAttribute):
            return DataQueryGroup(nodes=[self, self.__class__(attribute=other)])
        elif isinstance(other, self.__class__):
            return DataQueryGroup(nodes=[self, other])


@dataclass
class DataQueryGroup:

    def __init__(self, nodes: list[DataQueryField]):
        for node in nodes:
            if not isinstance(node, DataQueryField):
                raise
        self.nodes = nodes
        self._attributes = [node.attribute.fullname for node in self.nodes]

    @property
    def attributes(self):
        return sorted(self._attributes)

    @property
    def tree(self):
        tree = {}
        for node in self.nodes:
            attr = node.attribute
            if attr.parent not in tree:
                tree[attr.parent] = [attr.name]
            else:
                tree[attr.parent].append(attr.name)
        return tree

    def to_gql(self):
        gql = ''
        for parent, children in self.tree.items():
            if not parent:
                gql += ' '.join(child for child in children) + ' '
            else:
                gql += f'{parent}{{{" ".join(child for child in children)}}} '
        if gql[-1] == ' ':
            gql = gql[:-1]
        return gql

    @overload
    def __add__(self, other: DataAttribute) -> 'DataQueryGroup': ...

    @overload
    def __add__(self, other: DataQueryField) -> 'DataQueryGroup': ...

    @overload
    def __add__(self, other: 'DataQueryGroup') -> 'DataQueryGroup': ...

    def __add__(self, other: DataAttribute or DataQueryField or 'DataQueryGroup') -> 'DataQueryGroup':
        if isinstance(other, DataAttribute):
            if other.fullname not in self._attributes:
                self.nodes.append(DataQueryField(attribute=other))
                self._attributes.append(other.fullname)
        elif isinstance(other, DataQueryField):
            if other.attribute.fullname not in self._attributes:
                self.nodes.append(other)
                self._attributes.append(other.attribute.fullname)
        elif isinstance(other, self.__class__):
            for node in other.nodes:
                attr = node.attribute
                if attr.fullname not in self._attributes:
                    self.nodes.append(node)
                    self._attributes.append(attr.fullname)
        return self

