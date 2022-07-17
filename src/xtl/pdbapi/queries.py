from .nodes import QueryField
from .operators import *


def has_uniprot_id(id_: str):
    '''
    Search for entries with a specific UniProt ID

    :param id_: UniProt ID to search for
    :return:
    '''
    f1 = QueryField(
        ExactMatchOperator(
            attribute='rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name',
            value='UniProt'
        )
    )

    f2 = QueryField(
        ExactMatchOperator(
            attribute='rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession',
            value=id_
        )
    )

    return f1 & f2

