from dataclasses import dataclass, Field
from typing import Any, Optional

from xtl.common import AnnotatedDataclass


@dataclass
class GPhLConfig(AnnotatedDataclass):

    def _get_alias_value(self, param: Field):
        """
        Format the value of a parameter using an `alias_fstring` metadata field and the values of other parameters
        specified in the `alias_fstring_keys` metadata field.
        """
        fstring = param.metadata.get('alias_fstring', None)
        fdict = {key: getattr(self, key) for key in param.metadata.get('alias_fstring_keys', [])}
        if None in fdict.values():
            return None
        return fstring.format(**fdict)

    def _get_param_from_alias(self, alias: str) -> Optional[Field]:
        """
        Get the parameter name from an alias.
        """
        for param in self.__dataclass_fields__.values():
            if param.metadata.get('alias', None) == alias:
                return param
        param = self.__dataclass_fields__.get(alias, None)
        return param

    def get_param_value(self, name: str) -> dict[str, Any]:
        """
        Get the value of a parameter as a dictionary in the format `{parameter: value}`.
        """
        param = self.__dataclass_fields__.get(name, None)
        if not param:
            raise ValueError(f'Invalid parameter {name}')

        # Parse compound parameters
        if param.metadata.get('param_type') == 'compound':
            value = {}
            for subparam in param.metadata.get('members', []):
                value.update(self.get_param_value(subparam))
            formatter = param.metadata.get('formatter', None)
            if formatter:
                value = formatter(value)
            return value

        value = getattr(self, name)

        p = param.metadata.get('alias', name)
        v = self._get_alias_value(param) if 'alias_fstring' in param.metadata else value

        formatter = param.metadata.get('formatter', None)
        if formatter:
            v = formatter(v)
        if isinstance(v, bool):
            v = 'yes' if v else 'no'
        if isinstance(v, str):
            v = f'"{v}"'
        return {p: v}

    def get_group(self, name: str) -> dict[str, Any]:
        """
        Get all parameters in a group as a dictionary.
        """
        results = {}
        for param in self.__dataclass_fields__.values():
            if param.metadata.get('group', None) == name:
                value = self.get_param_value(param.name)
                results.update(self.get_param_value(param.name))
        return results

    def get_all_params(self, modified_only: bool = False, grouped: bool = False) -> dict[str, Any]:
        """
        Get all the parameters in the configuration as a dictionary. If `modified_only` is set to `True`, only
        parameters with non-default values will be included. If `grouped` is set to `True`, the parameters will be
        returned in the groups specified in the `_groups` attribute.
        """
        results = {}
        if grouped and hasattr(self, '_groups'):
            for group, comment in self._groups.items():
                params = self.get_group(group)
                if modified_only:
                    new_params = {}
                    for k, v in params.items():
                        p = self._get_param_from_alias(k)
                        if p:
                            if not hasattr(p, 'default'):
                                continue
                            if v == p.default:
                                continue
                        new_params.update({k: v})
                    params = new_params
                if params:
                    results.update({group: {'comment': comment, 'params': params}})
        else:
            for param in self.__dataclass_fields__.values():
                if param.metadata.get('param_type', None) in ['__internal', 'compound']:
                    continue
                value = self.get_param_value(param.name)
                name = param.metadata.get('alias', param.name)
                if modified_only and value[name] == param.default:
                    continue
                results.update(value)
        return results
