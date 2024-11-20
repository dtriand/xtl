import platform


def get_os_name_and_version() -> str:
    if platform.system() == 'Linux':
        try:
            import distro
            return f'{distro.name()} {distro.version()}'
        except ModuleNotFoundError:
            return f'{platform.system()} {platform.version()}'
    else:
        return f'{platform.system()} {platform.version()}'