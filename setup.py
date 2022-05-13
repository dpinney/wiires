'''
Install wiires.
'''

from setuptools import setup

setup(
	name='wiires',
	version='1.0.0',
	packages=['wiires'],
	install_requires = open("requirements.txt").readlines(),
	include_package_data=True
)