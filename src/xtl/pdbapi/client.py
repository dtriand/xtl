import json
import requests
import warnings

from .options import ReturnType, RequestOptions
from .nodes import SearchQueryNode, SearchQueryField, SearchQueryGroup
from xtl.exceptions import InvalidArgument


class QueryResponse:

    def __init__(self, response: requests.Response):
        """
        Representation of an RCSB API response.

        :param response: requests.Response object to process
        """
        try:
            self.json = response.json()
        except:
            self.json = {}
        self.query_id = self.json.get('query_id', '')
        self.result_type = self.json.get('result_type', '')
        self.total_count = self.json.get('total_count', '')
        self.explain_meta_data = self.json.get('explain_meta_data', '')
        self.result_set = self.json.get('result_set', '')
        self.group_set = self.json.get('group_set', '')
        self.facets = self.json.get('facets', '')

    @property
    def pdbs(self) -> list[str]:
        """
        A list of the PDB IDs that resulted from a search.

        :return:
        """
        return [item['identifier'] for item in self.result_set]


class Client:

    SEARCH_URL: str = 'https://search.rcsb.org/rcsbsearch/v2/query'
    DATA_URL: str = 'https://data.rcsb.org/rest/v1/core'

    def __init__(self, request_options=RequestOptions()):
        '''
        A client for quering the RCSB Search API

        :param request_options:
        '''
        self.return_type = ReturnType.ENTRY
        self.request_options = request_options
        self._query: SearchQueryField or SearchQueryGroup

    @property
    def request(self):
        '''
        Request to send to the API

        :return:
        '''
        result = {
            'return_type': self.return_type.value,
        }
        if self._query:
            result['query'] = self._query.to_dict()
        if self.request_options.to_dict():
            result['request_options'] = self.request_options.to_dict()
        return result

    def search(self, query: SearchQueryField or SearchQueryGroup):
        '''
        Perform a query using the RCSB Search API

        :param query:
        :return:
        '''
        if not issubclass(query.__class__, SearchQueryNode):
            raise InvalidArgument(raiser='query', message='Must be QueryField or QueryGroup')
        self._query = query
        response = requests.post(url=Client.SEARCH_URL, data=json.dumps(self.request))

        if not response.ok:
            warnings.warn(f'It appears request failed with status {response.status_code}:\n{response.text}')
            response.raise_for_status()
        if response.status_code == 204:
            warnings.warn('Request processed successfully, but no hits were found (status: 204).')

        return QueryResponse(response)

    def data(self, query: list, schema: str = 'entry'):
        '''
        Perform a query using the RCSB Data REST API. Experimental implementation!

        :param query:
        :param schema:
        :return:
        '''
        warnings.warn('Experimental implementation of Client.data()')
        response = requests.get(url=f'{Client.DATA_URL}/{schema}/{"/".join(query)}')

        if not response.ok:
            warnings.warn(f'It appears request failed with status {response.status_code}:\n{response.text}')
            response.raise_for_status()
        if response.status_code == 204:
            warnings.warn('Request processed successfully, but no hits were found (status: 204).')

        return json.loads(response.text)
