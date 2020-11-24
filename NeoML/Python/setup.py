import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension

setup(
	name='neoml',
	version='1.0.1',
	description='Package Description',
	url='http://github.com/neoml-lib/neoml',  
	install_requires=['numpy>=1.19.1', 'scipy>=1.5.2'],
	include_package_data=True,
	packages=['neoml'],
	zip_safe=False,
	test_suite='tests'
)
