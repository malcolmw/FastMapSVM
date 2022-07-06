import setuptools

def configure():
# Initialize the setup kwargs
    kwargs = {
            "name": "fastmap",
            "version": "0.2a0",
            "author": "Malcolm White",
            "author_email": "malcolmw@mit.edu",
            "maintainer": "Malcolm White",
            "maintainer_email": "malcolmw@mit.edu",
            "url": "http://malcolmw.github.io/FastMapSVM",
            "description": "Prototype implementation of FastMapSVM algorithm for classifying complex objects.",
            "download_url": "https://github.com/malcolmw/FastMapSVM.git",
            "platforms": ["linux"],
            "requires": ["h5py", "numpy", "pandas", "scipy", "sklearn", "tqdm"],
            "packages": ["fastmap"]
            }
    return(kwargs)

if __name__ == "__main__":
    kwargs = configure()
    setuptools.setup(**kwargs)
