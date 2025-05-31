.. |Table| replace:: :class:`Table <xtl.common.tables.Table>`
.. |MissingValue| replace:: :class:`MissingValue <xtl.common.tables.MissingValue>`
.. |MissingValueConfig| replace:: :class:`MissingValueConfig <xtl.common.tables.MissingValueConfig>`
.. |numpy| replace:: :mod:`numpy`
.. |array| replace:: :func:`numpy.ndarray`
.. |pandas| replace:: :mod:`pandas`
.. |DataFrame| replace:: :class:`pandas.DataFrame <pandas.DataFrame>`
.. |rich| replace:: :mod:`rich`
.. |RichTable| replace:: :class:`rich.Table <rich.table.Table>`

Tables
======

The :mod:`xtl.common.tables` module provides the |Table| class for handling 2D tabular data. |Table| provides slicing
capabilities similar to |DataFrame|, handles serialization of missing values and supports various import/export options.

Quickstart
----------

The module provides three main classes: |Table|, |MissingValue| and |MissingValueConfig|.
Below is a simple example:

.. code-block:: python

   from xtl.common.tables import Table

   # Create a table with data and headers
   table = Table(
       data=[
           [1, 'Alpha', 10.5, True],
           [2, 'Beta', 20.3, False],
           [3, 'Gamma', 30.7, True]
       ],
       headers=['ID', 'Name', 'Value', 'Active']
   )

   # Access data using pandas-like slicing
   subset = table['Name':'Active', 1:3]  # Slice by column names and row indices
   single_value = table['Name', 2]  # Get the name in the third row: 'Gamma'

   # Modify data
   table['Value', 2] = 35.0  # Update a single value
   table['ID'] = [10, 20, 30]  # Update an entire column

   # Export to different formats
   csv_string = table.to_csv()
   pandas_df = table.to_pandas()
   numpy_array = table.to_numpy()
   rich_table = table.to_rich()  # For console printing

Missing values are handled automatically:

.. code-block:: python

   table = Table(
       data=[
           ['A', 1, 'x'],
           ['B', 'N/A', 'y'],
           ['C', 3, None]
       ],
       headers=['Col1', 'Col2', 'Col3'],
       missing_values=['N/A', None],  # Treat these as missing
       missing_value_repr='-'         # And represent them as '-'
   )

   print(table)
   # Col1 | Col2 | Col3
   # -----+------+-----
   # A    | 1    | x
   # B    | -    | y
   # C    | 3    | -

Core Concepts
-------------

Table initialization
^^^^^^^^^^^^^^^^^^^^

A |Table| can be initialized in several ways:

1. Directly with data and headers:

   .. code-block:: python

      table = Table(
          data=[[1, 'A'], [2, 'B'], [3, 'C']],
          headers=['Number', 'Letter']
      )

2. Empty table with predefined headers:

   .. code-block:: python

      # Create an empty table first, then add rows one by one
      table = Table(headers=['Name', 'Age', 'City'])
      table.add_row(['John', 30, 'New York'])
      table.add_row(['Alice', 25, 'London'])

3. From other data structures:

   .. code-block:: python

      # Create from a dictionary with column names as keys
      table = Table.from_dict({
          'Name': ['John', 'Alice', 'Bob'],
          'Age': [30, 25, 35],
          'City': ['New York', 'London', 'Paris']
      })

      # Create from a numpy array with optional headers
      import numpy as np
      array = np.array([[1, 2, 3], [4, 5, 6]])
      table = Table.from_numpy(array, headers=['A', 'B', 'C'])

      # Create from a pandas DataFrame
      import pandas as pd
      df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
      table = Table.from_pandas(df)


Appending data
^^^^^^^^^^^^^^

Data can easily be appended to an existing |Table| using either dedicated methods or Python built-in operations:

.. code-block:: python

   table = Table(
          data=[[1, 'A'], [2, 'B']],
          headers=['Number', 'Letter'],
          missing_values=[None],
          missing_value_repr='?'
      )

   # Append a row
   table.add_row([3, 'C'])
   table += [4, 'D']       # Using the += operator
   table += {'Number': 0}  # Only appends the 'Number' column, 'Letter' will be MissingValue
   print(table)
   # Number | Letter
   # -------+-------
   # 1      | A
   # 2      | B
   # 3      | C
   # 4      | D
   # 0      | ?

   # Append a column
   table.add_col(['a', 'b', 'c', 'd', None], col_name='Lowercase')
   table |= {'IsEven': [False, True, False, True, True]}  # Using the |= operator
   table.add_col('Empty')
   print(table)
   # Number | Letter | Lowercase | IsEven | Empty
   # -------+--------+-----------+--------+------
   # 1      | A      | a         | False  | ?
   # 2      | B      | b         | True   | ?
   # 3      | C      | c         | False  | ?
   # 4      | D      | d         | True   | ?
   # 0      | ?      | ?         | True   | ?

Removing data
^^^^^^^^^^^^^

Similarly, data can be removed from a |Table| using methods or Python built-in operations:

.. code-block:: python

   # Remove a row
   table.del_row(4)
   print(table)
   # Number | Letter | Lowercase | IsEven | Empty
   # -------+--------+-----------+--------+------
   # 1      | A      | a         | False  | ?
   # 2      | B      | b         | True   | ?
   # 3      | C      | c         | False  | ?
   # 4      | D      | d         | True   | ?

   # Remove a column
   table.del_col('Lowercase')    # By name
   table.del_col(0)              # By index
   table -= ['IsEven', 'Empty']  # Using the -= operator
   print(table)
   # Letter
   # ------
   # A
   # B
   # C
   # D

Accessing data
^^^^^^^^^^^^^^

|Table| implements a slicing syntax inspired by |DataFrame|, allowing slicing either by name or by index:

.. code-block:: python

   table = Table(
       data=[
           [1, 'Alpha', 10.5, True],
           [2, 'Beta', 20.3, False],
           [3, 'Gamma', 30.7, True]
       ],
       headers=['ID', 'Name', 'Value', 'Active']
   )

   # Get a single column as a Table
   table['Name']

   # Get multiple columns
   table['ID':'Value']  # Columns from 'ID' through 'Value'
   table[0:3]  # First three columns

   # Get a single cell
   table['Name', 2]  # Value at 'Name' column, row index 2
   table[1, 2]  # Value at column index 1, row index 2

   # Get a subset of rows from a column
   table['Value', 1:3]  # Values from rows 1, 2 in 'Value' column

   # Get a rectangular subset
   table['Name':'Active', 1:3]  # Columns 'Name' through 'Active', rows 1, 2

   # Using negative indices (from the end)
   table[-1]  # Last column
   table['Name', -1]  # Last row in 'Name' column

   # Using step in slices
   table[::2]  # Every other column
   table[:, ::2]  # Every other row

You can also set values using similar syntax:

.. code-block:: python

   # Set a single cell
   table['Value', 2] = 35.0

   # Set multiple cells in a column
   table['Value', 1:3] = [25.0, 35.0]

   # Set an entire column
   table['Active'] = [True, True, False]

.. note::

   |Table| does not provide any type of data validation or type checking. It only serves as a structured data container.

Alternatively, |Table| also support methods for accessing and modifying data:

.. code-block:: python

   # Get a row by index
   table.get_row(1)  # Returns the second row

   # Get a column by name or index
   table.get_col('Name')  # By name
   table.get_col(2)  # By index (0-based)

   # Set data for a row
   table.set_row(0, [10, 'Alpha Prime', 15.0, True])

   # Set data for a column
   table.set_col('Active', [False, True, True])

Missing values
^^^^^^^^^^^^^^

|Table| provides support for handling missing values through |MissingValue| and |MissingValueConfig|. Values that should
be treated as missing and their serialization are defined during table initialization:

.. code-block:: python

   # Create a table with potentially missing data
   table = Table(
       data=[
           [1, 'Alpha', 10.5],
           [2, 'Beta', 'N/A'],
           [3, 'Gamma', None]
       ],
       headers=['ID', 'Name', 'Value'],
       missing_values=['N/A', None],  # Values to treat as missing
       missing_value_repr='-'         # Representation for missing values
   )

When you access the data, missing values are returned using the specified representation:

.. code-block:: python

   print(table.data)
   # [[1, 'Alpha', 10.5], [2, 'Beta', '-'], [3, 'Gamma', '-']]

   # Original values are preserved internally
   print(table._data)
   # [[1, 'Alpha', 10.5], [2, 'Beta', MissingValue('N/A')], [3, 'Gamma', MissingValue(None)]]

You can update the missing values configuration at any time, and all values will update accordingly:

.. code-block:: python

   # Change which values are considered missing
   table.missing = ['N/A']  # We no longer treat 'None' as missing

   # Change the representation
   table.missing.repr = '?'

   # All values have been re-evaluated
   print(table)
   # ID | Name  | Value
   # ---+-------+------
   # 1  | Alpha | 10.5
   # 2  | Beta  | ?
   # 3  | Gamma | None

Table operations
^^^^^^^^^^^^^^^^

|Table| supports operations for combining tables:

.. code-block:: python

    # Create two tables with the same structure
    table1 = Table(
        data=[[1, 'A'], [2, 'B']],
        headers=['Number', 'Letter']
    )

    table2 = Table(
        data=[[3, 'C'], [4, 'D']],
        headers=['Number', 'Letter']
    )

    # Concatenate rows (tables must have the same columns)
    combined = table1 + table2

    # In-place row concatenation
    table1 += table2

    # Table with different columns
    table3 = Table(
        data=[[True, 10], [False, 20]],
        headers=['Active', 'Value']
    )

    # Concatenate columns (tables must have the same number of rows)
    wider_table = table1 | table3

    # In-place column concatenation
    table1 |= table3

I/O operations
--------------

|Table| supports various import and export formats.

Data import
^^^^^^^^^^^

Import from Python built-in types:

.. code-block:: python

   # Create from a dictionary with column names as keys
   data = {
       'ID': [1, 2, 3],
       'Name': ['Alpha', 'Beta', 'Gamma'],
       'Value': [10.5, '-', '-']
   }
   table = Table.from_dict(data_dict)

Import from third-party libraries:

.. code-block:: python

   # Import from a numpy array
   import numpy as np
   array = np.array([[1, 'Alpha', 10.5], [2, 'Beta', 'N/A'], [3, 'Gamma', None]])
   table = Table.from_numpy(array, headers=['ID', 'Name', 'Value'])

   # Import from a pandas DataFrame
   import pandas as pd
   df = pd.DataFrame({
       'ID': [1, 2, 3],
       'Name': ['Alpha', 'Beta', 'Gamma'],
       'Value': [10.5, '-', '-']
   })
   table = Table.from_pandas(df)

Import from CSV:

.. code-block:: python

   # From a file
   table = Table.from_csv('data.csv', header_line=0)

   # From a string
   csv_content = """ID,Name,Value
   1,Alpha,10.5
   2,Beta,20.3
   3,Gamma,30.7"""
   table = Table.from_csv(csv_content, header_line=0)

   # With custom parameters
   table = Table.from_csv('data.csv',
                         delimiter=';',         # Use semicolon as delimiter
                         header_line=None,      # No header line
                         header_char='#',       # Headers prefixed with # character
                         new_line='\r\n')       # Custom newline character

Data export
^^^^^^^^^^^

Export to Python built-in types:

.. code-block:: python

   table = Table(
      data=[
          [1, 'Alpha', 10.5],
          [2, 'Beta', 'N/A'],
          [3, 'Gamma', None]
      ],
      headers=['ID', 'Name', 'Value'],
      missing_values=['N/A', None],  # Values to treat as missing
      missing_value_repr='-'         # Representation for missing values
   )

   # Get raw data as nested lists
   data = table.data
   # [[1, 'Alpha', 10.5], [2, 'Beta', '-'], [3, 'Gamma', '-']]

   # Get all values as a flattened list
   flat_list = table.to_list()
   # [1, 'Alpha', 10.5, 2, 'Beta', '-', 3, 'Gamma', '-']

   # Get dictionary with column names as keys
   data_dict = table.to_dict()
   # {'ID': [1, 2, 3], 'Name': ['Alpha', 'Beta', 'Gamma'], 'Value': [10.5, '-', '-']}

Export to third-party libraries:

.. code-block:: python

   # Export to numpy array
   numpy_array = table.to_numpy()

   # Export to pandas DataFrame
   pandas_df = table.to_pandas()

   # Export to rich.Table for console display
   rich_table = table.to_rich()

   # Custom formatting can be applied when creating a rich table
   rich_table = table.to_rich(cast_as=lambda v: f"Value: {v}")

Export to CSV:

.. code-block:: python

   # Get as CSV string
   csv_string = table.to_csv()

   # Write to CSV file
   table.to_csv(filename='data.csv')

.. code-block:: csv
   :caption: ``data.csv``

   ID,Name,Value
   1,Alpha,10.5
   2,Beta,-
   3,Gamma,-

But the CSV export can also be customized with various parameters:

.. code-block:: python

   # Customize CSV output with various parameters
   table.to_csv(
       filename='data.csv',
       delimiter=';',
       new_line='\r\n',
       header_char='# ',  # Add a character before the header line
   )

.. code-block:: csv
   :caption: ``data.csv``

   # ID;Name;Value
   1;Alpha;10.5
   2;Beta;-
   3;Gamma;-

Console
^^^^^^^

|Table| can be printed directly into the console:

.. code-block:: python

   print(table)
   # Number | Letter | Active | Value
   # -------+--------+--------+------
   # 1      | A      | True   | 10
   # 2      | B      | False  | 20
   # 3      | C      | True   | 30
   # 4      | D      | False  | 40

For more advanced terminal display, you can use the :func:`to_rich() <xtl.common.tables.Table.to_rich>` method:

.. code-block:: python

   from rich.console import Console

   console = Console()
   rich_table = table.to_rich()
   console.print(rich_table)
