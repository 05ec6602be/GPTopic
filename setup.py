from setuptools import setup, find_packages


with open("README.md", 'r', encoding='utf') as f:
    long_description = f.read()

setup(
    name='gptopic',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gensim',
        'hdbscan',
        'nltk',
        'numpy', 
        'openai==0.28.1',
        'pandas',
        'plotly',
        'regex',
        'scikit-learn',
        'seaborn',
        'sentence-transformers',
        'tiktoken',
        'tokenizers',
        'tqdm',
        'umap-learn',
        'umap-learn[plot]'
        ],
    include_package_data=True,
    # Additional metadata
    author='Anonymous',
    author_email='anonymous',
    description='A package for integrating LLMs like GPT-3.5 and GPT-4 into topic modelling',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords=['Topic Modelling', 'GPT', 'LLM', 'OpenAI', 'Retrieval Augmented Generation', 'Chat-GPT', 'GPT-3', 'GPT-4'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Science/Research',
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

