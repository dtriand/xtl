import requests

from .attributes import Attribute, AttributeGroup


class RCSBSchema:

    def __init__(self, verbose=False):
        self._verbose = verbose
        self.schema_version: str
        self.unparsable_objects = []
        self._schema_url = 'http://search.rcsb.org/rcsbsearch/v2/metadata/schema'
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
            attr = Attribute(name=attr_name, type=obj['type'], description=description)
            setattr(self, attr_name, attr)
            return attr
        elif obj['type'] == 'array':
            self._turn_object_to_attribute(attr_name=attr_name, obj=obj['items'])
        elif obj['type'] == 'object':
            group = AttributeGroup()
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

