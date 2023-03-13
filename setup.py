from setuptools import setup, find_packages
setup(
    name = 'm2dtIceSheet',
    version = '0.0.1',
    description = 'Multifaceted Mathematics for Predictive Digital Twins for Ice Sheet Application',
    url = '',
    author = 'Mohan Krishnamoorthy',
    author_email = 'mkrishnamoorthy2425@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        'numpy>=1.15.0',
        'scipy>=1.5.4',
        'matplotlib>=3.0.0',
        # 'mpi4py>=3.0.0',
        # 'pandas>=1.1.5',
    ],
    scripts=[],
    extras_require = {
    },
    entry_points = {
    },
    dependency_links = [
    ]
)
