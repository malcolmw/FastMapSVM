import re
import setuptools

version_file = 'fastmap/_version.py'
version_line = open(version_file, 'r').read()
version_re = r'^__version__ = ["\']([^"\']*)["\']'
mo = re.search(version_re, version_line, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError(f'Unable to find version string in {version_file}.')

def configure():
# Initialize the setup kwargs
    kwargs = {
            'name': 'FastMap',
            'version': version,
            'author': 'Malcolm C. A. White',
            'author_email': 'malcolmw@mit.edu',
            'maintainer': 'Malcolm C. A. White',
            'maintainer_email': 'malcolmw@mit.edu',
            'url': 'http://malcolmw.github.io/FastMapSVM',
            'description': 'Official implementation of FastMapSVM algorithm '
                'for classifying complex objects (White et al., 2023).',
            'download_url': 'https://github.com/malcolmw/FastMapSVM.git',
            'platforms': ['linux'],
            'install_requires': ['numpy', 'tqdm'],
            'packages': ['fastmap']
            }
    return kwargs

if __name__ == '__main__':
    kwargs = configure()
    setuptools.setup(**kwargs)
