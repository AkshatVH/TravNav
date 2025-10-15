from setuptools import setup

package_name = 'py_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rocon',
    maintainer_email='szilard.molnar@aut.utcluj.ro',
    description='This package is meant to receive geometry_msgs and calculate the distance from the Twist information.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'listener = py_pubsub.twist_subscriber_distance_calculator:main',
            'listener_isaac = py_pubsub.twist_subscriber_distance_calculator_isaac:main',
        ],
        
    },
)
