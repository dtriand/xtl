from pygments.lexer import RegexLexer, bygroups
from pygments.token import Keyword, Literal, Name, Operator, Punctuation


class CsvLexer(RegexLexer):
    # Simple CSV lexer based on: https://stackoverflow.com/a/25508711/298171

    name = 'Csv'
    aliases = ['csv', 'comma-separated', 'comma-separated-values']
    filenames = ['*.csv']

    tokens = {
        'root': [
            (r'^[^,\n]*', Operator, 'second'),
        ],
        'second': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Name.Constant), 'third'),
        ],
        'third': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Keyword.Declaration), 'fourth'),
        ],
        'fourth': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Literal.Number), 'fifth'),
        ],
        'fifth': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Literal.String.Single), 'sixth'),
        ],
        'sixth': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Name.Constant), 'seventh'),
        ],
        'seventh': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Keyword.Namespace), 'eighth'),
        ],
        'eighth': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Literal.Number), 'ninth'),
        ],
        'ninth': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Literal.String.Single), 'tenth'),
        ],
        'tenth': [
            (r'(,)([^,\n]*)', bygroups(Punctuation, Keyword.Type), 'unsupported'),
        ],
        'unsupported': [
            (r'(.+)', bygroups(Punctuation)),
        ],
    }