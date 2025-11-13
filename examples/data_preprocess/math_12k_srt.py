# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess DAPO dataset to parquet format

This file can be used to create an unlabeled version of the same dataset,
to be used for self-rewarding training (SRT).
"""

import os
import datasets
import numpy as np
from verl.utils.hdfs_io import copy, makedirs
import argparse
from sympy.parsing.latex import parse_latex

GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED = "LABEL_BY_SELF_CONSISTENCY"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--stage_splits", type=lambda x: [int(i) for i in x.split(',')], default=[30, 40, 30])
    parser.add_argument("--stage_noises", type=lambda x: [float(i) for i in x.split(',')], default=[0.0, 1.0, 1.0])
    parser.add_argument("--add_self_consistency_labels", action='store_true')
    parser.add_argument('--dataset_path', type=str, default='hiyouga/math12k')

    args = parser.parse_args()

    data_source = 'math_12k'
    stage_splits = args.stage_splits
    stage_noises = args.stage_noises

    dataset_path = args.dataset_path
    dataset = datasets.load_dataset(dataset_path, trust_remote_code=True)
    dataset = dataset.shuffle(seed=42)
    # split the dataset into stage splits by percentage
    stage_train_datasets = []
    # Calculate cumulative percentages and create splits
    cumulative_percent = 0
    for i, split_percent in enumerate(stage_splits):
        start_idx = int(len(dataset['train']) * cumulative_percent / 100)
        cumulative_percent += split_percent
        end_idx = int(len(dataset['train']) * cumulative_percent / 100) if i < len(stage_splits) - 1 else len(dataset['train'])
        stage_train_datasets.append(dataset['train'].select(range(start_idx, end_idx)))
    
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def evaluate_latex(solution):
        sym = parse_latex(solution)
        val = sym.evalf()
        return str(int(val))

    def is_latex_expression(solution):
        return "\\" in solution
    # add a row to each data item that represents a unique id
    def make_map_fn_train(split, label_noise):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            original_solution = example.pop('answer')
            # check if solution is a latex expression
            # if is_latex_expression(original_solution):
            #     original_solution = evaluate_latex(original_solution)
            # else:
            original_solution = str((original_solution))

            # add noise
            random_num = np.random.uniform(low=0.0, high=1.0, size=None)

            if random_num <= label_noise:
                print("Original solution: ", original_solution)

                if args.add_self_consistency_labels:
                    solution = GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED
                else: 
                    solution = str(int(original_solution)+1)
                    
                print("Label noise solution: ", solution)

            else:
                solution = original_solution

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                    "solution_hidden_during_training": original_solution,
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return lambda example, idx: process_fn(example=example, idx=idx)
    
    def make_map_fn_test(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            solution = example.pop('answer')
            solution = str(solution)

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                    "solution_hidden_during_training": solution,
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn


    for i in range(len(stage_splits)):
        print(f"Processing stage {i} with noise {stage_noises[i]} and split {stage_splits[i]}")
        stage_train_dataset = stage_train_datasets[i]
        stage_test_dataset = test_dataset
        stage_train_dataset = stage_train_dataset.map(
            function=make_map_fn_train('train', label_noise=stage_noises[i]), 
            with_indices=True,
        )
        stage_test_dataset = stage_test_dataset.map(
            function=make_map_fn_test('test'), with_indices=True,
        )

        os.makedirs(os.path.join(args.local_dir, f'stage_{i}'), exist_ok=True)
        stage_train_dataset.to_parquet(os.path.join(args.local_dir, f'stage_{i}', f'train.parquet'))
        stage_test_dataset.to_parquet(os.path.join(args.local_dir, f'stage_{i}', 'test.parquet'))

        if args.hdfs_dir is not None:
            makedirs(os.path.join(args.hdfs_dir, f'stage_{i}'), exist_ok=True)
            copy(src=os.path.join(args.local_dir, f'stage_{i}', 'train.parquet'), dst=os.path.join(args.hdfs_dir, f'stage_{i}', 'train.parquet'))
            copy(src=os.path.join(args.local_dir, f'stage_{i}', 'test.parquet'), dst=os.path.join(args.hdfs_dir, f'stage_{i}', 'test.parquet'))