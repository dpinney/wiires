'''
Install wiires.
'''

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

def pre_install():
    check_call("pip install git+https://github.com/wind-python/windpowerlib.git@60f58ce76555b3ff33b31a959cd14958546b24e8".split())
    check_call("pip install git+https://github.com/oemof/feedinlib.git@cd5d2392b398953e40a38909ea0f3a94986cb632".split())

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
