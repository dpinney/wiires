'''
Install wiires.
'''

from setuptools import setup

setup(
	name='wiires',
	version='1.0.0',
	py_modules=['wiires'],
	install_requires = open("requirements.txt").readlines(),
	include_package_data=True
)
