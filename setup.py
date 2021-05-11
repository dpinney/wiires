'''
Install wiires.
'''

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

def pre_install():
    check_call("pip install git+https://github.com/wind-python/windpowerlib".split())
    check_call("git+https://github.com/oemof/feedinlib".split())

class PreDevelopCommand(develop):
    def run(self):
        pre_install()
        develop.run(self)

class PreInstallCommand(install):
    def run(self):
        pre_install()
        install.run(self)

setup(
	name='wiires',
	version='1.0.0',
	py_modules=['LCEM','dss_manipulation','graph_dss','scipy_optimize','wiires'],
	install_requires = open("requirements.txt").readlines(),
	cmdclass={
		'develop':PreDevelopCommand,
		'install':PreInstallCommand
	}
)
