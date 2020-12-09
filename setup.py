from setuptools import setup, find_packages
import os
import shutil


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    """
    Create a list of packages from requirements.txt.
    Pop `setuptools` which is added by PyCharm.
    :return:
    """
    with open('requirements.txt') as f:
        req = f.read().splitlines()
    for r in req:
        if r.startswith('setuptools'):
            req.remove(r)
    return req


if os.path.exists('xtl.egg-info'):
    shutil.rmtree('xtl.egg-info')

setup(
    name='xtl',
    version='0.0.0',
    license='GPLv3',
    url='https://github.com/dtriand/xtl',
    download_url='',
    project_urls={
        'Bug Tracker': '',
        'Documentation': 'https://xtl.readthedocs.io'
    },
    author='Dimitris Triandafillidis',
    author_email='dimitristriandafillidis@gmail.com',
    description='',
    long_description=readme(),
    keywords='',
    package_dir={"": "src"},
    packages=find_packages(where='src', exclude=['tests', 'tests.*']),  # Do not include tests in distribution
    python_requires='>=3.7',
    install_requires=requirements(),
    # extras_require={}  # Specified in setup.cfg
    #     # Optional requirements. Install via pip install xtl[interactive]
    # },
    tests_require=['pytest'],
    package_data={
        # '': ['requirements.txt']
        # non-py files to include
        # specified in MANIFEST.in instead
    },
    entry_points={
        'console_scripts': [
            'xtl = xtl.cli.cli:cli_main',
            'gsas2 = xtl.cli.cli_gsas2:cli_gsas'
        ]
    }
)

