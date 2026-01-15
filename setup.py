from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "pygame"
    "opencv-python"
]

# Installation operation
setup(
    name="go2w_vtm",
    packages=["go2w_vtm"],
    author="albusgive",
    maintainer="albusgive",
    url="",
    version="0.0.1",
    description="",
    keywords="",
    install_requires=INSTALL_REQUIRES,
    license="Apache License 2.0",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
