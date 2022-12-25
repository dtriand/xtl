import pytest

from xtl.pdbapi.schema import _RCSBSchema
from xtl.pdbapi.attributes import _Attribute, _AttributeGroup


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
