from setuptools import setup, find_namespace_packages

setup(
    name='PDS_Project',
    version='0.0.1dev1',
    description='Semester Project - Programming Data Science',
    author='Student',
    author_email='student@uni-koeln.de',
    packages=find_namespace_packages(include=['*']),
    install_requires=['pandas', 'geopandas', 'shapely', 'numpy', 'scikit-learn', 'click', 'yaspin'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.cli:cli']
    }
)
