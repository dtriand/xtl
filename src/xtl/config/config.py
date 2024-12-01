from .. import __version__
from ..exceptions import FileError, InvalidArgument

import os
import copy
import configupdater

# 'section': {
#     'option': 'value $ restraints # comment'
# }
# restraint is a python expression, where x is the value (e.g. '0 < x < 5')

# Default config dictionary. Used to determine base sections/options/values
# Private, should not be accessed
_cfg = {
    'xtl': {
        'version': __version__
    },
    'automate': {
        'compute_site': 'local',
        'change_permissions': 'False $ x in ["True", "False"] # Change permissions of files and directories after execution of external jobs',
        'permissions_files': '700 # Permissions for files',
        'permissions_directories': '700 # Permissions for directories',
    },
    'units': {
        'temperature': 'C $ x in "CKF" # C for Celsius, K for Kelvin, F for Fahrenheit'
    },
    'dependencies': {
        'gsas': '# Path for GSAS2 installation'
    },
    'cli': {
        'rich_output': 'True $ x in ["True", "False"] # Enable rich output',
        'striped_table_rows': 'True $ x in ["True", "False"] # Alternating row colors in tables',
        'gsas_instprm_template_dir': '# Directories for holding template .instprm files'
    }
}

# Copy of default config. Used to construct config file.
# Public, can be accessed to add new default entries (e.g. registering plugin entries)
cfg = copy.deepcopy(_cfg)


class _ValueRestraintComment:

    # Delimiters
    del_comment = '#'
    del_restraint = '$'

    def __init__(self, text):
        """
        Split a `text` string to a value, restraint and comment, based on whether they contain restraint and comment
        delimiters, `$` and `#` respectively.

        :param str text:
        """
        self._text = text
        self.value = ''
        self.restraint = ''
        self.comment = ''

        if self.del_comment not in self._text and self.del_restraint not in self._text:
            # 'option': 'value'
            self.value = self._text
        elif self.del_comment in self._text and self.del_restraint not in self._text:
            # 'option': 'value # comment'
            value, comment = self._text.split(self.del_comment)
            self.value = value.rstrip()
            self.comment = comment.lstrip()
        elif self.del_comment not in self._text and self.del_restraint in self._text:
            # 'option': 'value $ restraint'
            value, restraint = self._text.split(self.del_restraint)
            self.value = value.rstrip()
            self.restraint = restraint.lstrip()
        else:
            # 'option': 'value $ restraint # comment
            non_comment, comment = self._text.split(self.del_comment)
            value, restraint = non_comment.split(self.del_restraint)
            self.value = value.rstrip()
            self.restraint = restraint.strip()
            self.comment = comment.lstrip()


class DefaultConfig:

    def __init__(self, cfg=cfg):
        self.cfg = cfg
        self.config = {}
        for section, options in self.cfg.items():
            opts = {}
            if isinstance(options, dict):
                for option, content in options.items():
                    vrc = _ValueRestraintComment(content)
                    opts[option] = {
                        'value': vrc.value,
                        'restraint': vrc.restraint,
                        'comment': vrc.comment
                    }
            elif isinstance(options, str):
                opts = 'None'
            self.config[section] = opts
            self.__setattr__(section, opts)

    def __getattr__(self, item):
        return self.item


class Config(configupdater.ConfigUpdater):

    def __init__(self, file, fix=False):
        """

        :param str file: file to read
        :param bool fix: try to fix possible errors in file
        :raises FileError: if error in file
        """
        self._file = file
        self.fix = fix

        super().__init__()

        if os.path.exists(self._file):
            self.read(self._file)
            error = self.validate()
            if error:
                if self.fix:
                    print(f'Attempting to fix errors in config file: {self._file}')
                    self.upgrade_config()
                    error = self.validate()
                    if error:
                        raise FileError(file=self._file, message='Invalid config file.', details=error)
                else:
                    raise FileError(file=self._file, message='Invalid config file.', details=error)
        else:
            self.generate_config()

    def generate_config(self, default_config=DefaultConfig()):
        """
        Appends all sections/options/values in ``default_config`` to self and then saves config to file. If a comment is
        available for an option, it will be appended before the option.

        :param DefaultConfig default_config:
        :return:
        """

        for section, options in default_config.config.items():
            self.add_section(section)
            if options == 'None':
                pass
            elif isinstance(options, dict):
                pointer = self[section]['placeholder'] = 'placeholder'
                for i, (option, content) in enumerate(options.items()):
                    comment = content['comment']
                    value = content['value']
                    if i == 0:
                        if comment:
                            pointer = self[section]['placeholder'].add_after.comment(comment).option(option, value)
                        else:
                            pointer = self[section]['placeholder'].add_after.option(option, value)
                    else:
                        if comment:
                            pointer = pointer.comment(comment).option(option, value)
                        else:
                            pointer = pointer.option(option, value)
            self.remove_option(section, 'placeholder')
            self[section].add_after.space(1)
        self.save()

    def validate(self):
        """
        Checks if all sections and option in :class:`DefaultConfig` are available in the config. Values are also
        validated if restraints are in place.

        :return: None if no error else error
        :rtype: None or str
        """
        default_cfg = DefaultConfig().config
        for section, options in default_cfg.items():
            if section not in self.sections():
                return f"Missing section '{section}'"
            if isinstance(options, dict):
                for option, content in options.items():
                    if option not in self[section]:
                        return f"Missing option '{option}' in section '{section}'"
                    value = self[section][option].value
                    restraint = content['restraint']
                    x = value  # x is the variable in each restraint
                    if restraint:  # empty restraints cannot be evaluated
                        try:
                            if not eval(restraint):  # if evaluates to False
                                return f"Invalid value for option [{section}][{option}] = {value}. " \
                                       f"The following expression must be true: {restraint}"
                        except TypeError:
                            x = eval(value)  # attempt to convert string to number
                            if not eval(restraint):  # reevaluate restrain
                                return f"Invalid value for option [{section}][{option}] = {value}. " \
                                       f"The following expression must be true: {restraint}"
        return None
    
    def save(self):
        # Save changes to file
        if self._filename is None:
            with open(self._file, 'w') as f:
                self.write(f)
        else:
            self.update_file()

    def upgrade_config(self):
        # Update config file with new sections and options from _default_config
        # Don't overwrite user values with default values
        updated = DefaultConfig()
        updated_cfg = updated.config
        for section in self.sections():
            if section not in updated_cfg:
                pass
            elif section in updated_cfg:
                for option in self.options(section):
                    if option not in updated_cfg[section]:
                        continue
                    elif option in updated_cfg[section]:
                        if section != 'xtl' and option != 'version':
                            if not self.fix:
                                updated_cfg[section][option]['value'] = self[section][option].value
                            else:
                                self.fix = False
                                x = self[section][option].value
                                restraint = updated_cfg[section][option]['restraint']
                                if restraint and eval(restraint):  # if not empty restraint and evaluates to True
                                    updated_cfg[section][option]['value'] = self[section][option].value
                                elif not updated_cfg[section][option]['value']:  # No default value for option
                                    updated_cfg[section][option]['value'] = self[section][option].value
                                else:  # Keep default value
                                    pass
            self.remove_section(section)
        os.remove(self._file)
        self._filename = None
        self.generate_config(updated)

    def restore_config(self):
        """
        Deletes config file and generates a new one based on :class:`DefaultConfig`. All values are restored to
        defaults. Any sections/options added with :meth:`Config.register_options()` are also dropped.

        :return:
        """
        for section in self.sections():
            self.remove_section(section)
        os.remove(self._file)
        self.generate_config(default_config=DefaultConfig(cfg=_cfg))

    def register_options(self, entry):
        if not isinstance(entry, dict):
            raise InvalidArgument(raiser='entry', message='Must be type dict')
        for section, options in entry.items():
            if section in _cfg:
                # Do not add options in default sections
                continue
            else:
                cfg[section] = {}
                if section not in self:
                    self.add_section(section)
            if isinstance(options, dict):
                for option, content in options.items():
                    cfg[section][option] = content
                    # self[section][option] = _ValueRestraintComment(content).value  # Bug: Overwrites user entries
            if not options:
                cfg.pop(section)
                self.remove_section(section)
        self.upgrade_config()

