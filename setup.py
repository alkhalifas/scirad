from setuptools import setup, find_packages

setup(
    name="scirad",
    version="0.1.0",
    description="A scientific research and summarization tool.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your_username/scirad",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,  # Includes files specified in MANIFEST.in
    install_requires=[
        "requests",
        "mlflow",
        "streamlit",
        "langchain",
        "tiktoken",
        "nltk",
        "numexpr",
        # Add any other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
