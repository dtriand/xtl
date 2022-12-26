import pytest

import requests

from xtl.pdbapi.schema import _RCSBSchema, SearchSchema, DataSchema
from xtl.pdbapi.attributes import _Attribute, _AttributeGroup
from xtl.pdbapi.data.options import DataService


def mock_get_schema_json(self):
    return {
        'type': 'object',
        'properties': {
            'category1': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'item1': {
                            'type': 'integer',
                            'description': ''
                        },
                        'item2': {
                            'type': 'string',
                            'description': ''
                        }
                    },
                    'additionalProperties': False,
                    'required': []
                }
            },
            'category2': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'item1': {
                            'type': 'integer',
                            'description': ''
                        },
                        'subcategory1': {
                            'type': 'object',
                            'properties': {
                                'subsubcategory1': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'object',
                                        'properties': {
                                            'subsubitem1': {
                                                'type': 'string',
                                                'description': ''
                                            },
                                            'subsubitem2': {
                                                'type': 'float',
                                                'description': ''
                                            }
                                        }
                                    }
                                },
                                'subitem1': {
                                    'type': 'string',
                                    'description': ''
                                }
                            },
                            'additionalProperties': False
                        }
                    },
                    'additionalProperties': False,
                    'required': []
                }
            }
        },
        'additionalProperties': False,
        'required': [],
        '$schema': '',
        'title': '',
        'description': '',
        '$comment': 'Schema version: 0.0.0'
    }


class _DummySchema(_RCSBSchema):

    def __init__(self, verbose=False):
        self._AttributeCls = _Attribute
        self._AttributeGroupCls = _AttributeGroup
        self._schema_url = ''
        super().__init__(verbose=verbose)


class TestRCSBSchema:

    def test_turn_object_to_attribute(self, mocker):
        mocker.patch('xtl.pdbapi.schema._RCSBSchema._get_schema_json', mock_get_schema_json)
        attr = _DummySchema()

        c1 = getattr(attr, 'category1', None)
        assert c1
        assert c1._children == ['item1', 'item2']

        i1 = getattr(attr, 'category1.item1', None)
        assert i1
        assert i1.parent == 'category1'

        c2 = getattr(attr, 'category2', None)
        assert c2
        assert c2._children == ['item1', 'subcategory1']

        c2sc1 = getattr(attr, 'category2.subcategory1', None)
        assert c2sc1
        assert c2sc1.parent_name == 'category2'
        assert c2sc1.parent == c2
        assert c2sc1._children == ['subsubcategory1', 'subitem1']

        c2sc1si1 = getattr(attr, 'category2.subcategory1.subitem1', None)
        assert c2sc1si1
        assert c2sc1si1.name == 'subitem1'
        assert c2sc1si1.parent == 'category2.subcategory1'

        c2sc1ssc1 = getattr(attr, 'category2.subcategory1.subsubcategory1', None)
        assert c2sc1ssc1
        assert c2sc1ssc1.parent_name == 'category2.subcategory1'
        assert c2sc1ssc1.parent == c2sc1
        assert c2sc1ssc1._children == ['subsubitem1', 'subsubitem2']

        c2sc1ssc1ssi1 = getattr(attr, 'category2.subcategory1.subsubcategory1.subsubitem1', None)
        assert c2sc1ssc1ssi1
        assert c2sc1ssc1ssi1.name == 'subsubitem1'
        assert c2sc1ssc1ssi1.parent == 'category2.subcategory1.subsubcategory1'

        assert attr.schema_version == '0.0.0'

@pytest.mark.requires_network
class TestURLs:

    def test_search_schema_urls(self):
        assert requests.head(SearchSchema._base_url).ok

    @pytest.mark.parametrize('service', [DataService.ENTRY, DataService.POLYMER_ENTITY, DataService.BRANCHED_ENTITY,
                                         DataService.NON_POLYMER_ENTITY, DataService.POLYMER_INSTANCE,
                                         DataService.BRANCHED_INSTANCE, DataService.NON_POLYMER_INSTANCE,
                                         DataService.ASSEMBLY, DataService.CHEMICAL_COMPONENT])
    def test_data_schema_urls(self, service):
        # Data API does not support HEAD requests
        assert requests.options(f'{DataSchema._base_url}/{service.value}').ok


@pytest.mark.requires_network
class TestSearchSchema:
    attr = SearchSchema()

    def test_schema_version(self):
        assert tuple(int(i) for i in self.attr.schema_version.split('.')) >= (1, 36, 0)


@pytest.mark.requires_network
@pytest.mark.parametrize('service, schema_version', [(DataService.ENTRY, (9, 0, 0)),
                                                     (DataService.POLYMER_ENTITY, (10, 0, 0)),
                                                     (DataService.BRANCHED_ENTITY, (10, 0, 0)),
                                                     (DataService.NON_POLYMER_ENTITY, (10, 0, 0)),
                                                     (DataService.POLYMER_INSTANCE, (10, 0, 0)),
                                                     (DataService.BRANCHED_INSTANCE, (9, 0, 0)),
                                                     (DataService.NON_POLYMER_INSTANCE, (10, 0, 0)),
                                                     (DataService.ASSEMBLY, (9, 0, 0)),
                                                     (DataService.CHEMICAL_COMPONENT, (7, 1, 2))])
class TestDataSchema:

    def test_schema_version(self, service, schema_version):
        attr = DataSchema(service=service)
        assert tuple(int(i) for i in attr.schema_version.split('.')) >= schema_version