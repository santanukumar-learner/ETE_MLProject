# from setuptools import find_packages, setup
# from typing import List

# hypen='-e .'

# def get_requrements(file_path:str) -> List[str]:
#     requirements=[]
#     with open(file_path) as file_obj:
#         requirements=file_obj.readlines()
#         [req.replace('\n', " ") for req in requirements]
#         if hyper in requiremets:
#             requirements.remove(hyper)
#     print(requirements)

# setup(
#     name='mlproject',
#     version='0.0.1',
#     author='santanu',
#     author_email="santanukumaradhikari07@gmail.com",
#     packages=find_packages(),
#     install_requires=get_requirements('REQUIREMENTS.txt')

# )

from setuptools import find_packages, setup
from typing import List

hyphen = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if hyphen in requirements:
            requirements.remove(hyphen)
    # return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='santanu',
    author_email="santanukumaradhikari07@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('REQUIREMENTS.txt')
)