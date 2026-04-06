JOB_STYLES: dict[str, str] = {
    'xtl.jobs.jobs.Job.job_id': 'blue',
    'xtl.jobs.pool.BasePool.pool_id': 'magenta',
}
"""Style for formatting job and pool log messages."""

STYLES: dict[str, str] = {
    **JOB_STYLES,
}
"""Aggregated style-sheet for XTL"""
