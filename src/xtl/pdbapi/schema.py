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
        self.schema_version: str
        self.unparsable_objects = []

        self._schema_url: str
        if not hasattr(self, '_schema_url'):
            raise Exception('No schema URL provided.')

        self._parse_schema()

    def _get_schema_json(self):
        '''
        Download the RCSB schema as a json/dict object

        :return:
        '''
        response = requests.get(self._schema_url)
        response.raise_for_status()
        return response.json()

    def _turn_object_to_attribute(self, attr_name: str, obj: dict):
        '''
        Convert an dict object to an instance attribute

        :param attr_name: attribute name
        :param obj: dict to parse
        :return:
        '''
        if 'type' not in obj:
            # Temporary workaround for objects with multiple types ('anyOf' instead of 'type')
            # such as 'rcsb_branced_instance_feature.additional_properties.values'
            self.unparsable_objects.append(attr_name)
            if self._verbose:
                print(f'Unparsable object: {attr_name}')
            return

        if obj['type'] in ('string', 'number', 'integer', 'float'):
            description = obj['description'].replace('\n', ' ') if 'description' in obj else ''
            attr = self._AttributeCls(name=attr_name, type=obj['type'], description=description)
            setattr(self, attr_name, attr)
            return attr
        elif obj['type'] == 'array':
            self._turn_object_to_attribute(attr_name=attr_name, obj=obj['items'])
        elif obj['type'] == 'object':
            group = self._AttributeGroupCls()
            for child_attr_name, child_obj in obj['properties'].items():
                child_attr_fullname = f'{attr_name}.{child_attr_name}' if attr_name else child_attr_name
                child_group = self._turn_object_to_attribute(attr_name=child_attr_fullname, obj=child_obj)
                setattr(group, child_attr_name, child_group)
            setattr(self, attr_name, group)
            return group
        else:
            raise TypeError(f'Unrecognised node type {obj["type"]!r} of {attr_name}')

    def _parse_schema(self):
        '''
        Download schema and set all properties as instance attributes

        :return:
        '''
        json = self._get_schema_json()
        self.schema_version = json.get('$comment', 'Schema version: ').replace('Schema version: ', '')
        return self._turn_object_to_attribute('', json)


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

        self._schema_url = f'https://data.rcsb.org/rest/v1/schema/{self._data_service.value}'

        super().__init__(verbose=verbose)
