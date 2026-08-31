import itertools
from dataset_generation.multi_simairr import make_random_seed
from dataset_generation.phase_2_q2.experimental_data.generate_cmv_metadata_q2 import generate_cmv_metadata_files


## usage example #
# if __name__ == '__main__':
#     params = itertools.product([(400, 200, 6000), (400, 200, 12000), (400, 200, 25000)], [1, 2, 3])
#     previous_seeds = []
#     for (train_size, test_size, depth), run_number in params:
#         print(train_size, test_size, depth, run_number)
#         seed = make_random_seed(previous_seeds)
#         generate_cmv_metadata_files(
#             "/path/to/original_CMV_metadata.csv",
#             "/path/to/output_dir_for_writing_training_and_test_metadata_files",
#             "q3", "sequencing_depth",
#             run_number, train_size, test_size, depth, seed)
