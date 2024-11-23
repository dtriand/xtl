import os
import gemmi
import numpy as np
import tabulate


class mmCIF:

    def __init__(self, gemmi_doc, debug=False):
        """

        :param gemmi.cif.Document gemmi_doc:
        :param bool debug:
        """
        if not isinstance(gemmi_doc, gemmi.cif.Document):
            raise
        self.doc = gemmi_doc
        self.debug = debug

    def pretty_export(self, filename):
        import json
        mmjson_string = self.doc.as_json(mmjson=True)
        if self.debug:
            with open(f'{os.path.splitext(filename)[0]}.json', 'w') as j:
                j.write(mmjson_string)
        mmjson = json.loads(mmjson_string)

        output = ''
        for data_block in mmjson:
            output += f'{data_block}\n#\n'
            for category in mmjson[data_block]:
                is_loop_category = False
                first_group_of_values = next(iter(mmjson[data_block][category].values()))
                if len(first_group_of_values) > 1:
                    is_loop_category = True
                if is_loop_category:
                    output += 'loop_\n'
                    values_list = []
                    for item in mmjson[data_block][category]:
                        output += f'_{category}.{item}\n'
                        values_list += [mmjson[data_block][category][item]]
                    output += self._loop_to_str(values_list)
                    output += '\n'
                else:
                    longest_length = self._longest_length(category, mmjson[data_block])
                    for item in mmjson[data_block][category]:
                        category_item = f'_{category}.{item}'.ljust(longest_length, ' ')
                        value = self._format_value(mmjson[data_block][category][item][0])
                        output += f'{category_item}   {value}\n'
                output += '#\n'

        with open(filename, 'w') as m:
            m.write(output)

    @staticmethod
    def _loop_to_str(values_list):
        array = np.array(values_list, dtype='object')
        array = array.swapaxes(0, 1)

        # Tabulate formatting settings
        tabulate._table_formats['plain'] = tabulate.TableFormat(
            lineabove=None,
            linebelowheader=None,
            linebetweenrows=None,
            linebelow=None,
            headerrow=tabulate.DataRow("", "  ", ""),
            datarow=tabulate.DataRow("", "   ", ""),  # 3 spaces between columns
            padding=0,
            with_header_hide=None,
        )
        return tabulate.tabulate(array, tablefmt='plain', floatfmt=".4f", colalign=('right',),
                                 missingval='?', showindex=False)

    @staticmethod
    def _longest_length(category, dictionary):
        """
        Returns the length of the longest item (no. of characters) in a dictionary.

        :param str category:
        :param dict dictionary: Must contain 'category'
        :return:
        """
        longest = max([f'_{category}.{item}' for item in dictionary[category]], key=len)
        return len(longest)

    @staticmethod
    def _format_value(value):
        # ToDo: Missing or NaN values
        if isinstance(value, str) and ' ' in value:
            value = gemmi.cif.quote(value)
        return value
