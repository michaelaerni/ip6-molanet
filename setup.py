from setuptools import setup, find_packages

# Load readme
with open("README.md") as readme_file:
    readme_text = readme_file.read()

# Load license
with open("LICENSE") as license_file:
    license_text = license_file.read()

# Load requirements
with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(
    name="molanet",
    version="0.0.1",
    description="Molanet skin segmentation network",
    long_description=readme_text,
    author="Michael Aerni, Patrick Del Conte",
    license=license_text,
    packages=find_packages(exclude=("tests", "docs", "notebooks")),
    install_requires=requirements
)
