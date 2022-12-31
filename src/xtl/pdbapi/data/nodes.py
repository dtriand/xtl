from dataclasses import dataclass
from typing import overload

from xtl.pdbapi.attributes import DataAttribute, DataAttributeGroup
from xtl.pdbapi.data.options import DataService
from xtl.exceptions import InvalidArgument


@dataclass
class DataQueryField:

    def __init__(self, attribute: DataAttribute):
        if not isinstance(attribute, DataAttribute):
            raise InvalidArgument(raiser='attribute', message=f'Must be of type \'DataAttribute\' '
                                                              f'not {type(attribute)}')
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
        for i, node in enumerate(nodes):
            if not isinstance(node, DataQueryField):
                raise InvalidArgument(raiser=f'nodes[{i}]', message=f'Must be of type \'DataQueryField\' not '
                                                                    f'\'{type(nodes[i])}\'')
        self._data_service: DataService = nodes[0].attribute._schema._data_service
        for i, node in enumerate(nodes):
            if node.attribute._schema._data_service != self._data_service:
                raise InvalidArgument(raiser=f'nodes[{i}]', message=f'DataQueryGroup already initialized as with '
                                                                    f'data_service={self._data_service.value}. Cannot '
                                                                    f'add a node from a different data_service ('
                                                                    f'nodes[{i}]->'
                                                                    f'{node.attribute._schema._data_service})')
        self.nodes = nodes
        self._attributes = [node.attribute.fullname for node in self.nodes]

    @property
    def data_service(self) -> str:
        """
        Data service used to query RCSB API.
        :return:
        """
        return self._data_service.value

    @property
    def attributes(self) -> list[str]:
        """
        A list of the stored attributes in alphabetical order
        :return:
        """
        return sorted(self._attributes)

    @property
    def tree(self) -> dict:
        """
        A dict representation of the GraphQL query.
        :return:
        """
        tree = {}
        for node in self.nodes:
            attr = node.attribute
            if attr.parent not in tree:
                tree[attr.parent] = [attr.name]
            else:
                tree[attr.parent].append(attr.name)
        return tree

    def to_gql(self) -> str:
        """
        Prepare a GraphQL query representing the contents of the stored attributes.
        :return:
        """
        gql = ''
        for parent, children in self.tree.items():
            if not parent:
                gql += ' '.join(child for child in children) + ' '
            else:
                gql += f'{parent}{{{" ".join(child for child in children)}}} '
        if gql[-1] == ' ':
            gql = gql[:-1]
        return gql

    @classmethod
    def from_object(cls, obj: DataAttribute or DataAttributeGroup or DataQueryField):
        """
        Build a DataQueryGroup instance from a DataAttribute, DataAttributeGroup or DataQueryField instance.
        :param obj:
        :return:
        """
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, DataQueryField):
            return cls(nodes=[obj])
        elif isinstance(obj, DataAttribute):
            return cls(nodes=[DataQueryField(attribute=obj)])
        elif isinstance(obj, DataAttributeGroup):
            return cls(nodes=[DataQueryField(attribute=child) for child in obj.attributes])
        else:
            raise InvalidArgument(raiser='obj', message='Must be of type \'DataAttribute\', \'DataAttributeGroup\' or '
                                                        '\'DataQueryField\'.')

    @overload
    def __add__(self, other: DataAttribute) -> 'DataQueryGroup': ...

    @overload
    def __add__(self, other: DataQueryField) -> 'DataQueryGroup': ...

    @overload
    def __add__(self, other: 'DataQueryGroup') -> 'DataQueryGroup': ...

    def __add__(self, other: DataAttribute or DataQueryField or 'DataQueryGroup') -> 'DataQueryGroup':
        if isinstance(other, DataAttribute):
            if other._schema.data_service != self.data_service:
                raise InvalidArgument(raiser='other', message=f'Cannot append DataAttribute with data_service='
                                                              f'\'{other._schema.data_service}\' to an '
                                                              f'instance of DataQueryGroup with data_service='
                                                              f'\'{other.data_service=}\'')
            if other.fullname not in self._attributes:
                self.nodes.append(DataQueryField(attribute=other))
                self._attributes.append(other.fullname)
        elif isinstance(other, DataQueryField):
            if other.attribute._schema.data_service != self.data_service:
                raise InvalidArgument(raiser='other', message=f'Cannot append DataQueryField with data_service='
                                                              f'\'{other.attribute._schema.data_service}\' to an '
                                                              f'instance of DataQueryGroup with data_service='
                                                              f'\'{other.data_service=}\'')
            if other.attribute.fullname not in self._attributes:
                self.nodes.append(other)
                self._attributes.append(other.attribute.fullname)
        elif isinstance(other, self.__class__):
            if other.data_service != self.data_service:
                raise InvalidArgument(raiser='other', message=f'Cannot append DataQueryGroup instances of different '
                                                              f'data services. '
                                                              f'{self.data_service=} {other.data_service=}')
            for node in other.nodes:
                attr = node.attribute
                if attr.fullname not in self._attributes:
                    self.nodes.append(node)
                    self._attributes.append(attr.fullname)
        return self

