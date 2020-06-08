from setuptools import setup, find_namespace_packages

setup(
    name='PDS_Project',
    version='1.0',
    description='Semester Project - Programming in Data Science',
    author='Lukas Humpe, Tim Schaefer, Michael The-Lan Bui, Philipp Page',
    author_email='lhumpe@smail.uni-koeln.de, tschaefer@smail.uni-koeln.de,mbui@smail.uni-koeln.de,'
                 'ppagejem@smail.uni-koeln.de',
    packages=find_namespace_packages(include=['*']),
    install_requires=['pandas', 'geopandas', 'shapely', 'numpy', 'scikit-learn', 'click', 'yaspin', 'joblib'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.cli:cli']
    }
)
