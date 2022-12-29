import requests

from xtl.pdbapi.attributes import SearchAttribute, SearchAttributeGroup, DataAttribute, DataAttributeGroup, \
    _Attribute, _AttributeGroup
from xtl.pdbapi.data.options import DataService
from xtl.exceptions import InvalidArgument


class _RCSBSchema:

    def __init__(self, verbose=False):
        self._AttributeCls: SearchAttribute or DataAttribute
        self._AttributeGroupCls: SearchAttributeGroup or DataAttributeGroup
        if (not hasattr(self, '_AttributeCls')) or (not hasattr(self, '_AttributeGroupCls')):
            raise Exception('Uninitialized instance.')

        self._verbose = verbose
        self._post_parsing_attributes = []
        self._post_parsing_groups = []
        self.schema_version: str
        self._dubious_entries = []
        self._attributes = []
        self._attribute_groups = []

        self._schema_url: str
        if not hasattr(self, '_schema_url'):
            raise Exception('No schema URL provided.')

        self._parse_schema()

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
            self._post_parsing_groups += [item_name]
            return True
        return False

    def _turn_object_to_attribute(self, attr_name: str, obj: dict, parent_attr_name: str = ''):
        """
        Convert an dict object to an instance attribute

        :param attr_name: attribute name
        :param obj: dict to parse
        :return:
        """

        # Objects that contain 'anyOf' instead of a type. Usually arrays.
        if 'type' not in obj and 'anyOf' in obj:
            description = obj['description'].replace('\n', ' ') if 'description' in obj else ''
            attr = self._AttributeCls(fullname=attr_name, type='array?', description=description)
            self._dubious_entries.append(attr_name)
            return attr

        # Standard objects with a single type
        if obj['type'] in ('string', 'number', 'integer', 'float'):
            description = obj['description'].replace('\n', ' ') if 'description' in obj else ''

            # Extract the parent group name
            name_tree = attr_name.rsplit('.', maxsplit=1)
            try:
                parent, name = name_tree
            except ValueError:
                parent, name = '', attr_name

            # Create _Attribute
            attr = self._AttributeCls(fullname=attr_name, type=obj['type'], description=description, name=name,
                                      parent=parent)

            # Associate _Attribute to _RCSBSchema and vice versa
            setattr(self, attr_name, attr)
            setattr(attr, '_schema', self)
            self._attributes.append(attr_name)  # Append the attribute's fullname for easy retrieval
            return attr
        elif obj['type'] == 'array':
            # Most arrays contain objects, hence attr will be an _AttributeGroup instance
            attr = self._turn_object_to_attribute(attr_name=attr_name, obj=obj['items'])

            # Rare cases: All of these return _Attribute instances
            if ('type' not in obj['items']) and ('anyOf' in obj['items']):  # Arrays containing 'anyOf'
                # Extract the parent group name
                name_tree = attr_name.rsplit('.', maxsplit=1)
                try:
                    parent, name = name_tree
                except ValueError:
                    parent, name = '', attr_name
                # Update attribute's properties
                attr.name = name
                attr.parent = parent
                attr.type = 'array'
                attr.contains = [i['type'] for i in obj['items']['anyOf']]
                self._post_parsing_attributes.append(attr)
            elif obj['items']['type'] != 'object':  # Arrays containing e.g. 'string' instead of 'object'
                attr.type = 'array'                     # this will be an _Attribute of type='array' that
                attr.contains = [obj['items']['type']]  # contains e.g. 'string'
                # Ensure that this attribute will be associated with the parent group after parsing completes
                self._post_parsing_attributes.append(attr)
            return attr
        elif obj['type'] == 'object':
            # Create dummy _AttributeGroup instance
            group = self._AttributeGroupCls(name_='')

            # Iterate over the nested items (i.e. 'properties' in the json file)
            for child_attr_name, child_obj in obj['properties'].items():
                # Build the attribute's fullname
                child_attr_fullname = f'{attr_name}.{child_attr_name}' if attr_name else child_attr_name
                # Parse the child's object. This returns either a _Attribute or a _AttributeGroup instance
                child = self._turn_object_to_attribute(attr_name=child_attr_fullname, obj=child_obj,
                                                       parent_attr_name=attr_name)
                # Check if the child's fullname contains a dot and also append that instance to list for postprocessing
                if self._is_tiered_item(child_attr_fullname) and child is not None:
                    # Set the parent for the child
                    if isinstance(child, self._AttributeCls):
                        child.parent = child_attr_fullname.rsplit('.', maxsplit=1)[0]
                    elif isinstance(child, self._AttributeGroupCls):
                        child.parent_name = child.name_.rsplit('.', maxsplit=1)[0]
                # Associate child with the parent
                setattr(group, child_attr_name, child)

            # Fix group's properties
            group.name_ = attr_name
            group.update_children()  # Create a list of children's full names
            if self._is_tiered_item(group.name_):  # Check if group has a parent and append to list for postprocessing
                group.parent_name = group.name_.rsplit('.', maxsplit=1)[0]

            # Associate group with schema and vice versa
            setattr(self, attr_name, group)
            setattr(group, '_schema', self)

            # Append group's fullname for easy retrieval
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
        self._turn_object_to_attribute('', json)
        self._post_parsing_cleanup()
        return

    def _post_parsing_cleanup(self):
        # Associate a parent for nested groups
        for group_name in self._post_parsing_groups:
            group = getattr(self, group_name, None)
            if isinstance(group, self._AttributeGroupCls) and group.parent_name:
                group.parent = getattr(self, group.parent_name)
        self._post_parsing_groups = []  # free memory
        # Associate attributes in the form of e.g. array['string'] to their parent group
        for attr in self._post_parsing_attributes:
            parent = getattr(self, attr.parent)
            setattr(parent, attr.name, attr)
            setattr(self, attr.fullname, attr)
        self._post_parsing_attributes = []  # free memory

    @property
    def attributes(self) -> list[str]:
        """
        The fullnames of the stored attributes. Can be used to retrieve the Attribute instances using getattr().
        :return:
        """
        return self._attributes

    @property
    def attribute_groups(self):
        """
        The fullnames of the stored attribute groups. Can be used to retrieve the AttributeGroup instances using
        getattr().
        :return:
        """
        return self._attribute_groups

    @property
    def schema_url(self):
        """
        The URL that was used to retrieve the schema from RCSB.
        :return:
        """
        return self._schema_url

    # def __setattr__(self, key, value):
    #     # For debugging
    #     if issubclass(value.__class__, _Attribute):
    #         print(f'A  {value.fullname}')
    #     elif issubclass(value.__class__, _AttributeGroup):
    #         print(f'AG {value.name_}')
    #     elif key in ['_AttributeCls', '_AttributeGroupCls', '_verbose', '_attributes', '_attribute_groups',
    #                  '_post_parsing_attributes', '_post_parsing_groups']:
    #         pass
    #     else:
    #         print(f'UK {key=} {value=}')
    #     self.__dict__[key] = value


class SearchSchema(_RCSBSchema):
    _base_url = 'http://search.rcsb.org/rcsbsearch/v2/metadata/schema'

    def __init__(self, verbose=False):
        self._AttributeCls = SearchAttribute
        self._AttributeGroupCls = SearchAttributeGroup

        self._schema_url = self._base_url

        super().__init__(verbose=verbose)


class DataSchema(_RCSBSchema):
    _base_url = 'https://data.rcsb.org/rest/v1/schema'

    def __init__(self, service: DataService = DataService.ENTRY, verbose=False):
        self._AttributeCls = DataAttribute
        self._AttributeGroupCls = DataAttributeGroup

        self._data_service: DataService = service
        if not isinstance(self._data_service, DataService):
            raise InvalidArgument(raiser='service', message='Must be of type \'DataService\'')

        self._schema_url = f'{self._base_url}/{self.data_service}'

        super().__init__(verbose=verbose)

    @property
    def data_service(self):
        return self._data_service.value
