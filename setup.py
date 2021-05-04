'''
Install wiires.
'''

from setuptools import setup

setup(
	name='wiires',
	version='1.0.0',
	py_modules=['LCEM','dssmanipulation','graphdss','scipyoptimize','wiires'],
	install_requires = open("requirements.txt").readlines(),
)
