import reciprocalspaceship as rs
from reciprocalspaceship.dtypes.base import MTZDtype


mtz_types: dict[str, MTZDtype] = {}
"""Dictionary of MTZ column types to their corresponding rs.MTZDtype class"""

mtz_summary = rs.summarize_mtz_dtypes(print_summary=False)
for i in range(mtz_summary.shape[0]):
    # Columns: ['MTZ Code', 'Name', 'Class', 'Internal']
    cls_name = mtz_summary['Class'][i]
    mtz_type = mtz_summary['MTZ Code'][i]
    mtz_types[mtz_type] = getattr(rs.dtypes, cls_name)
