from rich.highlighter import RegexHighlighter, Highlighter
from rich.text import Text


class JobHighlighter(RegexHighlighter):
    """
    Highlighter for xtl.jobs.job.Job loggers.
    """
    base_style = 'xtl.jobs.jobs.Job.'
    highlights = [
        r'(?P<job_id>\[([^\]]+)\])',
    ]

    def highlight(self, text: Text) -> None:
        if 'Job' not in text.plain:
            return
        super().highlight(text)


class PoolHighlighter(RegexHighlighter):
    """
    Highlighter for xtl.jobs.pools.BasePool loggers.
    """
    base_style = 'xtl.jobs.pool.BasePool.'
    highlights = [
        r'(?P<pool_id>\[([^\]]+)\])',
    ]

    def highlight(self, text: Text) -> None:
        if 'Pool' not in text.plain:
            return
        super().highlight(text)
