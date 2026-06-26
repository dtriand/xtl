from typing import Any

from pydantic import ValidationError


class OptionsValidationError(ValueError):
    """
    Custom validation error for Pydantic models with structured errors and cleaner formatting.
    """

    def __init__(self, exc: ValidationError, model_name: str):
        self.model_name = model_name
        self.errors = self._extract_errors(exc)
        super().__init__(str(self))

    def _extract_errors(self, exc: ValidationError) -> list[dict[str, Any]]:
        errors = []
        for e in exc.errors(include_url=False):
            ctx = e.get('ctx', {})
            cause = ctx.get('error')
            if isinstance(cause, OptionsValidationError):
                # Prepend the outer loc to all inner errors
                outer = self._format_location(e['loc'])
                for e in cause.errors:
                    # Drop the inner model name
                    inner = e['loc'].split('.', 1)[1]
                    errors.append({
                        **e,
                        'loc': f'{outer}.{inner}',  # Re-prefix with outer model name
                    })
            else:
                errors.append({
                    'loc': self._format_location(e['loc']),
                    'msg': e['msg'],
                    'type': e['type'],
                    'input': e.get('input', None),
                })
        return errors

    def _format_location(self, loc: tuple) -> str:
        """
        Convert a location tuple to a string with dot notation, *e.g.* ``('tables', 0, 'data') -> 'tables[0].data'``.

        :param loc: Location tuple from Pydantic error
        :return: Formatted location string
        """
        parts = [self.model_name]
        for part in loc:
            if isinstance(part, int):
                parts[-1] += f'[{part}]'
            else:
                parts.append(str(part))
        return '.'.join(parts)

    def __str__(self) -> str:
        n = len(self.errors)
        lines = [f'{n} validation error{"s" if n > 1 else ""} for {self.model_name}']
        for e in self.errors:
            lines.append(f'  {e["loc"]}')
            lines.append(f'    {e["msg"]}')
            lines.append(f'    [type={e["type"]}, input_value={e["input"]}, input_type={type(e["input"]).__name__}]')
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.errors!r})'