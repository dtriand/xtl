from enum import Enum
from dataclasses import dataclass


class ReturnType(Enum):
    # https://search.rcsb.org/#return-type
    ENTRY = 'entry'
        # Returns a list of PDB IDs.
    ASSEMBLY = 'assembly'
        # Returns a list of PDB IDs appended with assembly IDs in the format of a [pdb_id]-[assembly_id], corresponding
        # to biological assemblies.
    POLYMER_ENTITY = 'polymer_entity'
        # Returns a list of PDB IDs appended with entity IDs in the format of a [pdb_id]_[entity_id], corresponding to
        # polymeric molecular entities.
    NON_POLYMER_ENTITY = 'non_polymer_entity'
        # Returns a list of PDB IDs appended with entity IDs in the format of a [pdb_id]_[entity_id], corresponding to
        # non-polymeric entities (or ligands).
    POLYMER_INSTANCE = 'polymer_instance'
        # Returns a list of PDB IDs appended with asym IDs in the format of a [pdb_id].[asym_id], corresponding to
        # instances of certain polymeric molecular entities, also known as chains. Note, that asym_id in the instance
        #  identifier corresponds to the _label_asym_id from the mmCIF schema (assigned by the PDB). It can differ from
        #  _auth_asym_id (selected by the author at the time of deposition).
    MOL_DEFINITION = 'mol_definition'
        # Returns a list of molecular definition identifiers that include:
        # - Chemical component entries identified by the alphanumeric code, COMP ID: e.g. ATP, ZN
        # - BIRD entries identified by BIRD ID, e.g. PRD_000154


class NodeType(Enum):
    # https://search.rcsb.org/#building-search-request
    TERMINAL = 'terminal'
        # Performs an atomic search operation, e.g. searches for a particular value in a particular field. Parameters
        # in the terminal query clause provide match criteria for finding relevant search hits. The set of parameters
        # differs for different search services.
    GROUP = 'group'
        # Wraps other terminal or group nodes and is used to combine multiple queries in a logical fashion.


class SearchService(Enum):
    # https://search.rcsb.org/#search-services
    TEXT = 'text'
        # Performs attribute searches against textual annotations associated with PDB structures.
        # Refer to https://search.rcsb.org/structure-search-attributes.html for a full list of annotations.
    TEXT_CHEM = 'text_chem'
        # Performs attribute searches against textual annotations associated with PDB molecular definitions.
        # Refer to https://search.rcsb.org/chemical-search-attributes.html for a full list of annotations.
    FULL_TEXT = 'full_text'
        # Performs unstructured searches against textual annotations associated with PDB structures or molecular
        # definitions. Unstructured search performs a full-text searches against multiple text attributes.
    SEQUENCE = 'sequence'
        # This service employs the MMseqs2 software (https://github.com/soedinglab/MMseqs2) and performs fast sequence
        # matching searches (BLAST-like) based on a user-provided FASTA sequence (with E-value or % Identity cutoffs).
        # Following searches are available:
        # - protein: search for protein sequences
        # - dna: search for DNA sequences
        # - rna: search for RNA sequences
    SEQMOTIF = 'seqmotif'
        # Performs short motif searches against nucleotide or protein sequences, using three different types of input
        # format:
        # - simple (e.g., CXCXXL)
        # - prosite (e.g., C-X-C-X(2)-[LIVMYFWC])
        # - regex (e.g., CXCX{2}[LIVMYFWC])
    STRUCTURE = 'structure'
        # Performs fast searches matching a global 3D shape of assemblies or chains of a given entry (identified by PDB
        # ID), in either strict (strict_shape_match) or relaxed (relaxed_shape_match) modes, using a BioZernike
        # descriptor strategy
    STRUCMOTIF = 'strucmotif'
        # Performs structure motif searches on all available PDB structures.
    CHEMICAL = 'chemical'
        # Enables queries of small-molecule constituents of PDB structures, based on chemical formula and chemical
        # structure. Both molecular formula and formula range searches are supported. Queries for matching and similar
        # chemical structures can be performed using SMILES and InChI descriptors as search targets.


class LogicalOperator(Enum):
    AND = 'and'
    OR = 'or'


class ComparisonType(Enum):
    # https://search.rcsb.org/#comparison-operators
    EQUALS = 'equals'
    GREATER = 'greater'
    GREATER_OR_EQUAL = 'greater_or_equal'
    LESS = 'less'
    LESS_OR_EQUAL = 'less_or_equal'


class SortingDirectionOption(Enum):
    # https://search.rcsb.org/#sorting
    DESC = 'desc'
    ASC = 'asc'


class ScoringStrategy(Enum):
    # https://search.rcsb.org/#scoring-strategy
    COMBINED = 'combined'
    TEXT = 'text'
    SEQUENCE = 'sequence'
    SEQMOTIF = 'seqmotif'
    STRUCTURE = 'structure'
    STRUCMOTIF = 'strucmotif'
    CHEMICAL = 'chemical'


@dataclass
class SortOptions:
    # https://search.rcsb.org/#sorting

    def __init__(self, sort_by='score', direction=SortingDirectionOption.DESC):
        self.sort_by = sort_by
        self.direction = direction

    def to_dict(self):
        return {
            'sort_by': self.sort_by,
            'direction': self.direction.value
        }


@dataclass
class PagerOptions:
    # https://search.rcsb.org/#pagination

    def __init__(self, start=0, rows=100):
        self.start = start
        self.rows = rows

    def to_dict(self):
        return {
            'start': self.start,
            'rows': self.rows
        }


@dataclass
class FacetsOptions:
    # https://search.rcsb.org/#using-facets
    # ToDo: Add proper support for faceted searches

    def __init__(self, name: str, attribute: str, aggregation_type='terms'):
        self.name = name
        self.attribute = attribute
        self.aggregation_type = aggregation_type

    def to_dict(self):
        return {
            'name': self.name,
            'attribute': self.attribute,
            'aggregation_type': self.aggregation_type
        }


@dataclass
class RequestOptions:

    def __init__(self, scoring_strategy: ScoringStrategy = None, sort_options: list[SortOptions] = [],
                 pager: PagerOptions = None, return_all_hits=False, return_counts=False,
                 facets: list[FacetsOptions] = []):
        '''
        Controls various aspects of the search request including pagination, sorting, scoring and faceting. If omitted,
        the default parameters for sorting, scoring and pagination will be applied.
        https://search.rcsb.org/index.html#building-search-request

        :param scoring_strategy:
        :param sort_options:
        :param pager:
        :param return_all_hits:
        :param return_counts:
        :param facets:
        '''
        self.scoring_strategy = scoring_strategy

        if sort_options:
            for sort_option in sort_options:
                if not isinstance(sort_option, SortOptions):
                    raise
        self.sort_options = sort_options

        self.pager = pager
        self.return_all_hits = return_all_hits
        self.return_counts = return_counts

        if facets:
            for facet in facets:
                if not isinstance(facet, FacetsOptions):
                    raise
        self.facets = facets

    def to_dict(self):
        # https://search.rcsb.org/index.html#return-count
        if self.return_counts:
            return {
                'return_counts': True
            }

        result = {}
        if self.scoring_strategy:
            result['scoring_strategy'] = self.scoring_strategy.value
        if self.return_all_hits:
            result['return_all_hits'] = True
        if self.sort_options:
            result['sort'] = [sort_option.to_dict() for sort_option in self.sort_options]
        if self.pager:
            result['pager'] = self.pager.to_dict()
        if self.facets:
            result['facets'] = [facet.to_dict() for facet in self.facets]
        return result
