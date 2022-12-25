import requests

from xtl.pdbapi.attributes import SearchAttribute, SearchAttributeGroup, DataAttribute, DataAttributeGroup
from xtl.pdbapi.data.options import DataService
from xtl.exceptions import InvalidArgument


class _RCSBSchema:

    def __init__(self, verbose=False):
        self._AttributeCls: SearchAttribute or DataAttribute
        self._AttributeGroupCls: SearchAttributeGroup or DataAttributeGroup
        if (not hasattr(self, '_AttributeCls')) or (not hasattr(self, '_AttributeGroupCls')):
            raise Exception('Uninitialized instance.')

        self._verbose = verbose
        self._tiered_groups = []
        self.schema_version: str
        self.unparsable_objects = []
        self._attributes = []
        self._attribute_groups = []

        self._schema_url: str
        if not hasattr(self, '_schema_url'):
            raise Exception('No schema URL provided.')

        self._parse_schema()
        for group_name in self._tiered_groups:
            group = getattr(self, group_name, None)
            if isinstance(group, self._AttributeGroupCls) and group.parent_name:
                group.parent = getattr(self, group.parent_name)

    def _get_schema_json(self):
        """
        Download the RCSB schema as a json/dict object

        :return:
        """
        response = requests.get(self._schema_url)
        response.raise_for_status()
        return response.json()

    def _is_tiered_item(self, item_name: str):
        tree = item_name.split('.')
        if len(tree) > 1:
            self._tiered_groups += [item_name]
            return True
        return False

    def _turn_object_to_attribute(self, attr_name: str, obj: dict):
        """
        Convert an dict object to an instance attribute

        :param attr_name: attribute name
        :param obj: dict to parse
        :return:
        """
        if 'type' not in obj:
            # Temporary workaround for objects with multiple types ('anyOf' instead of 'type')
            # such as 'rcsb_branced_instance_feature.additional_properties.values'
            self.unparsable_objects.append(attr_name)
            if self._verbose:
                print(f'Unparsable object: {attr_name}')
            return

        if obj['type'] in ('string', 'number', 'integer', 'float'):
            description = obj['description'].replace('\n', ' ') if 'description' in obj else ''
            name_tree = attr_name.rsplit('.', maxsplit=1)
            try:
                parent, name = name_tree
            except ValueError:
                parent, name = '', attr_name
            attr = self._AttributeCls(fullname=attr_name, type=obj['type'], description=description, name=name,
                                      parent=parent)
            setattr(self, attr_name, attr)
            self._attributes.append(attr_name)
            return attr
        elif obj['type'] == 'array':
            self._turn_object_to_attribute(attr_name=attr_name, obj=obj['items'])
        elif obj['type'] == 'object':
            group = self._AttributeGroupCls(name_='')
            for child_attr_name, child_obj in obj['properties'].items():
                child_attr_fullname = f'{attr_name}.{child_attr_name}' if attr_name else child_attr_name
                child_group = self._turn_object_to_attribute(attr_name=child_attr_fullname, obj=child_obj)
                if self._is_tiered_item(child_attr_fullname) and child_group is not None:
                    if isinstance(child_group, self._AttributeCls):
                        child_group.parent = child_attr_fullname.rsplit('.', maxsplit=1)[0]
                    elif isinstance(child_group, self._AttributeGroupCls):
                        child_group.parent_name = child_group.name_.rsplit('.', maxsplit=1)[0]
                setattr(group, child_attr_name, child_group)
            group.name_ = attr_name
            group.update_children()
            if self._is_tiered_item(group.name_):
                group.parent_name = group.name_.rsplit('.', maxsplit=1)[0]
            setattr(self, attr_name, group)
            if group.name_ != '':  # don't append master object
                self._attribute_groups.append(attr_name)
            return group
        else:
            raise TypeError(f'Unrecognised node type {obj["type"]!r} of {attr_name}')

    def _parse_schema(self):
        """
        Download schema and set all properties as instance attributes

        :return:
        """
        json = self._get_schema_json()
        try:
            # Search schema: "Schema version: X.X.X" / Data schema: "schema_version: X.X.X"
            self.schema_version = json.get('$comment', 'Schema version: ').split(': ')[-1]
        except:
            self.schema_version = 'UNK'
        return self._turn_object_to_attribute('', json)

    @property
    def attributes(self):
        return self._attributes

    @property
    def attribute_groups(self):
        return self._attribute_groups

    @property
    def schema_url(self):
        return self._schema_url


class SearchSchema(_RCSBSchema):

    def __init__(self, verbose=False):
        self._AttributeCls = SearchAttribute
        self._AttributeGroupCls = SearchAttributeGroup

        self._schema_url = 'http://search.rcsb.org/rcsbsearch/v2/metadata/schema'

        super().__init__(verbose=verbose)


class DataSchema(_RCSBSchema):

    def __init__(self, service: DataService = DataService.ENTRY, verbose=False):
        self._AttributeCls = DataAttribute
        self._AttributeGroupCls = DataAttributeGroup

        self._data_service: DataService = service
        if not isinstance(self._data_service, DataService):
            raise InvalidArgument(raiser='service', message='Must be of type \'DataService\'')

        self._schema_url = f'https://data.rcsb.org/rest/v1/schema/{self.data_service}'

        super().__init__(verbose=verbose)

    @property
    def data_service(self):
        return self._data_service.value
