import os
import glob
import numpy as np
import json
import pdb

import diffuser.utils as utils


DATASETS = [
	f'{env}-{buffer}-v2'
	for env in ['hopper', 'walker2d', 'halfcheetah']
	for buffer in ['medium-replay', 'medium', 'medium-expert']
]

LOGBASE = 'logs/pretrained/'
TRIAL = '*'
EXP_NAME = 'plans*/*'
verbose = False


def load_results(paths):
	'''
		paths : path to directory containing experiment trials
	'''
	scores = []
	for i, path in enumerate(sorted(paths)):
		score = load_result(path)
		if verbose: print(path, score)
		if score is None:
			# print(f'Skipping {path}')
			continue
		scores.append(score)

		suffix = path.split('/')[-1]
		# print(suffix, path, score)

	if len(scores) > 0:
		mean = np.mean(scores)
	else:
		mean = np.nan

	if len(scores) > 1:
		err = np.std(scores) / np.sqrt(len(scores))
	else:
		err = 0
	return mean, err, scores

def load_result(path):
	'''
		path : path to experiment directory; expects `rollout.json` to be in directory
	'''
	fullpath = os.path.join(path, 'rollout.json')

	if not os.path.exists(fullpath):
		return None

	results = json.load(open(fullpath, 'rb'))
	score = results['score'] * 100
	return score


#######################
######## setup ########
#######################


if __name__ == '__main__':

	class Parser(utils.Parser):
	    dataset: str = None

	args = Parser().parse_args()

	for dataset in ([args.dataset] if args.dataset else DATASETS):
		subdirs = sorted(glob.glob(os.path.join(LOGBASE, dataset, EXP_NAME)))

		for subdir in subdirs:
			reldir = subdir.split('/')[-1]
			paths = glob.glob(os.path.join(subdir, TRIAL))
			paths = sorted(paths)

			mean, err, scores = load_results(paths)
			if np.isnan(mean):
				continue
			path, name = os.path.split(subdir)
			print(f'{dataset.ljust(30)} | {name.ljust(50)} | {path.ljust(50)} | {len(scores)} scores \n    {mean:.1f} +/- {err:.2f}')
			if verbose:
				print(scores)
