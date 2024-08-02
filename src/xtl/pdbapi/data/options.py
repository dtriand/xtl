from enum import Enum
from dataclasses import dataclass


class DataService(Enum):
    # https://data.rcsb.org/#data-schema
    ENTRY = 'entry'
    POLYMER_ENTITY = 'polymer_entity'
    BRANCHED_ENTITY = 'branched_entity'
    NON_POLYMER_ENTITY = 'nonpolymer_entity'
    POLYMER_INSTANCE = 'polymer_entity_instance'
    BRANCHED_INSTANCE = 'branched_entity_instance'
    NON_POLYMER_INSTANCE = 'nonpolymer_entity_instance'
    ASSEMBLY = 'assembly'
    CHEMICAL_COMPONENT = 'chem_comp'


@dataclass
class _GQLField:
    name: str
    identifiers: str
    separator: str = ''


class GQLField(Enum):
    # The GraphQL API also supports 'interface', 'entry_group', 'polymer_entity_group' and 'group_provenance' which are
    # not supported here yet.
    # https://data.rcsb.org/graphql/index.html
    ENTRY = _GQLField(name='entries', identifiers='entry_ids')
    POLYMER_ENTITY = _GQLField(name='polymer_entities', identifiers='entity_ids', separator='_')
    BRANCHED_ENTITY = _GQLField(name='branched_entities', identifiers='entity_ids', separator='_')
    NON_POLYMER_ENTITY = _GQLField(name='nonpolymer_entities', identifiers='entity_ids', separator='_')
    POLYMER_INSTANCE = _GQLField(name='polymer_entity_instances', identifiers='instance_ids', separator='.')
    BRANCHED_INSTANCE = _GQLField(name='branched_entity_instances', identifiers='instance_ids', separator='.')
    NON_POLYMER_INSTANCE = _GQLField(name='nonpolymer_entity_instances', identifiers='instance_ids', separator='.')
    ASSEMBLY = _GQLField(name='assemblies', identifiers='assembly_ids', separator='-')
    CHEMICAL_COMPONENT = _GQLField(name='chem_comps', identifiers='comp_ids')
