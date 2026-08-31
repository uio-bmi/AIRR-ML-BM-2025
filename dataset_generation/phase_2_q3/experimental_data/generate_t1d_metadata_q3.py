import itertools
from dataset_generation.multi_simairr import make_random_seed
from dataset_generation.phase_2_q1.experimental_data.split_t1d_metadata_phase_2_q1 import generate_metadata_files

## usage example #
# if __name__ == '__main__':
#     params = itertools.product([(400, 400, 6000), (400, 400, 12000), (400, 400, 25000)], [1, 2, 3])
#     previous_seeds = []
#     for (train_size, test_size, depth), run_number in params:
#         print(train_size, test_size, depth, run_number)
#         seed = make_random_seed(previous_seeds)
#         generate_metadata_files("/path/to/q3_t1d_metadata_files", "cohort1_2_3_anonymized.csv", "q3",
#                                 "sequencing_depth", depth, run_number, train_size, test_size, seed)
