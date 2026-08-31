from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='AIRR-ML-BM-2025',
    version='0.1',
    packages=['dataset_generation'],
    url='',
    license='MIT',
    author='Chakravarthi Kanduri',
    author_email='chakra.kanduri@gmail.com',
    description='',
    include_package_data=True,
    zip_safe=False,
    entry_points={'console_scripts': [
                                      'multi_ligo=dataset_generation.multi_ligo:execute',
                                      'multi_simairr=dataset_generation.multi_simairr:execute',
                                      'multi_simairr_var=dataset_generation.multi_simairr_variations:execute',
                                      'process_ligo_output=dataset_generation.process_ligo_output:execute',
                                      'split_train_test_all_dirs=pilot_data_analysis.split_train_test'
                                      ':execute_on_multiple_dirs',
                                      'split_train_test_single_dir=pilot_data_analysis.split_train_test'
                                      ':execute_on_single_dir',
                                      'profile_ml=pilot_data_analysis.profile_ml:execute',
                                      'logistic_interpretability=pilot_data_analysis.logistic_interpretability:execute',
                                      'emerson_interpretability=pilot_data_analysis.emerson_interpretability:execute',
                                      ]
    }
)
