from setuptools import setup

setup(
    name='PDS_Project',
    version='0.0.1dev1',
    description='Semester Project - Programming Data Science',
    author='Student',
    author_email='student@uni-koeln.de',
    packages=['nextbike'],
    install_requires=['pandas', 'geopandas', 'scikit-learn', 'click'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.cli:main']
    }
)
