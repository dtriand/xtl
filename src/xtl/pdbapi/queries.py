from xtl.pdbapi.search.nodes import SearchQueryField
from xtl.pdbapi.search.operators import *
from xtl.pdbapi.search.options import SearchService

def has_uniprot_id(id_: str):
    '''
    Search for entries with a specific UniProt ID

    :param id_: UniProt ID to search for
    :return:
    '''
    f1 = SearchQueryField(
        ExactMatchOperator(
            attribute='rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name',
            value='UniProt'
        )
    )

    f2 = SearchQueryField(
        ExactMatchOperator(
            attribute='rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession',
            value=id_
        )
    )

    return f1 & f2


def free_text(text: str):
    """

    :param text:
    :return:
    """

    q = SearchQueryField(
        UnstructuredTextOperator(
            value=text
        ),
        service=SearchService.FULL_TEXT
    )

    return q
