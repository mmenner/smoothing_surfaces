from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='smurf',
    url='https://github.com/...',
    author='Marco Menner',
    author_email='marco.menner88@gmail.com',
    # Needed to actually package something
    packages=['smurf'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'gurobipy (opt)', 'cvxopt (opt)'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='private',
    description='smoothing',
)
