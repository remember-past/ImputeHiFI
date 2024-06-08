import setuptools

if __name__ == "__main__":
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    setuptools.setup(name="ImputeHiFI",version="0.1.3",install_requires=required,)
