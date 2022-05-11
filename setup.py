from setuptools import setup

install_requires = [
    'astutils',
    'numpy',
    'scipy',
    'matplotlib',
    'pydot',
    'torch',
    'setuptools']

def run_setup():
    setup(name='synthesis_based_repair',
      version='0.1',
      packages=['synthesis_based_repair'],
      package_dir={'synthesis_based_repair': 'synthesis_based_repair'}
      )

if __name__ == "__main__":
    run_setup()
