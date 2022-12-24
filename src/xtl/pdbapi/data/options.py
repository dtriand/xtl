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
