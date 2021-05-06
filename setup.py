'''
Install wiires.
'''

from setuptools import setup

setup(
	name='wiires',
	version='1.0.0',
	py_modules=['LCEM','dss_manipulation','graph_dss','scipy_optimize','wiires'],
	install_requires = open("requirements.txt").readlines(),
)
