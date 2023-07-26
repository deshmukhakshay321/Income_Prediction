from setuptools import find_packages,setup
from typing import List
HYPON_E_DOT="-e ."

def get_requirements(filepath:str)-> List[str]:
    requirements=[]
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[i.replace("\n","") for i in requirements]

        if HYPON_E_DOT in requirements:
            requirements.remove(HYPON_E_DOT)

setup(name="ML_Pipeline_Project",
      version='0.0.1',
      escription='Machine Learning Pipeline Project',
      author=' Akshay Deshmukh',
      author_email='deshmukhakshay321@gmail.com',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=find_packages(),
      install_requires=get_requirements("requirements.txt"))

      