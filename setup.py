from setuptools import setup

setup(
    name='nextbike',
    version='0.0.1dev1',
    description="Semester Project - Programming Data Science",
    author="Student",
    author_email="student@uni-koeln.de",
    packages=["nextbike"],
    install_requires=['pandas', 'scikit-learn', 'click'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.cli:main']
    }
)
