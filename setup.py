import sys
import warnings

# if setup tools is not installed, bootstrap it
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages


if sys.version_info < (2, 7):
    sys.exit("Python version >=2.7 required")


def extras_require(key=None):
    """Deal with extra requirements

    Parameters
    ----------
    key : string, optional
        if not none, returns the requirements only for the given option

    Returns
    -------
    dictionary of requirements
    if key is not None: list of requirements
    """
    req_dic = {'py'}
    req_dic = {'doc': ['sphinx==1.5', 'numpydoc>=0.6', 'alabaster',
                       'sphinxcontrib-httpdomain']
               }

    req_dic['livedoc'] = req_dic['doc'] + ['sphinx-autobuild>=0.5.2', ]

    req_dic = {'doc'}

    req_dic['all'] = set(sum((v for v in req_dic.values()), []))

    if key:
        return req_dic[key]
    else:
        return req_dic


install_requires = ['numpy', 'scipy', 'astropy', 'matplotlib', 'astroquery']

# entry points
# scripts
entry_points = {'console_scripts':
                ['quick_fit = stellarSEDfits.quick_fit:main']}

setup(
    # package description and version
    name="stellarSEDfits",
    version='1.0.0',
    author="Greg Zeiman",
    author_email="gregz@astro.as.utexas.edu",
    description="Stellar SED fitting code",
    long_description=open("README.md").read(),

    # custom test class

    # list of packages and data
    packages=find_packages(),
    # get from the MANIFEST.in file which extra file to include
    include_package_data=True,
    # don't zip when installing
    zip_safe=False,

    # entry points: creates vhc script upon installation
    entry_points=entry_points,
    # dependences
    setup_requires=[],
    install_requires=install_requires,
    extras_require=extras_require(),

    classifiers=["Development Status :: 3 - Alpha",
                 "Environment :: Console",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: GNU General Public License (GPL)",
                 "Operating System :: Unix",
                 "Programming Language :: Python :: 2.7",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 ]
)
