import itertools
from dataset_generation.multi_simairr import make_random_seed
from dataset_generation.phase_2_q1.experimental_data.split_t1d_metadata_phase_2_q1 import generate_metadata_files



## usage example #
# if __name__ == '__main__':
#     params = itertools.product([(100, 400), (200, 400), (400, 400)], [1, 2, 3])
#     previous_seeds = []
#     for (train_size, test_size), run_number in params:
#         seed = make_random_seed(previous_seeds)
#         generate_metadata_files("/path/to/q2_t1d_metadata_files", "cohort1_2_3_anonymized.csv", "q2",
#                                 "training_dataset_size", train_size, run_number, train_size, test_size, seed)
