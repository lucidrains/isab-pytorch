from setuptools import setup, find_packages

setup(
  name = 'isab-pytorch',
  packages = find_packages(),
  version = '0.2.1',
  license='MIT',
  description = 'Induced Set Attention Block - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/isab-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism'
  ],
  install_requires=[
    'torch',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
