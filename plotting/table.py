import numpy as np
import pdb

from plotting.locomotion_scores import (
	means as MEANS,
	errors as ERRORS,
)

ALGORITHM_STRINGS = {
	# 'Diffuser': 'Diffuser (Ours)',
}

BUFFER_STRINGS = {
	'medium-expert': 'Medium-Expert',
	'medium': 'Medium',
	'medium-replay': 'Medium-Replay',
}

ENVIRONMENT_STRINGS = {
	'halfcheetah': 'HalfCheetah',
	'hopper': 'Hopper',
	'walker2d': 'Walker2d',
	'ant': 'Ant',
}

SHOW_ERRORS = ['Diffuser']

COLOR_DEFS = {
	'tblue': '1F77B4',
	'tred': 'FF6961',
	'tgreen': '429E9D',
	# 'thighlight': '1F77B4',
	'thighlight': '000000',
}

COLORS = {
	## sequence modeling
	'DT': 'black',
	'TT': 'black',
	## model-free
	'CQL': 'black',
	'IQL': 'black',
	'Onestep': 'black',
	## model-based
	'MOReL': 'black',
	'MBOP': 'black',
	'Diffuser': 'black',
}


def get_mean(d):
	return np.mean(list(d.values()))

def get_result(algorithm, buffer, environment, version='v2'):
	key = f'{environment}-{buffer}-{version}'
	mean = MEANS[algorithm].get(key, '-')
	if algorithm in SHOW_ERRORS:
		error = ERRORS[algorithm].get(key)
		return (mean, error)
	else:
		return mean

def set_highlights(scores, c=0.95):
	means = [
		score
		if type(score) != tuple
		else score[0]
		for score in scores
	]
	cutoff = max(means) * c
	highlights = [mean >= cutoff for mean in means]
	return highlights

def maybe_highlight(x, highlight):
	if highlight:
		return f'\\textbf{{\\color{{thighlight}}{x}}}'
	else:
		return x

def format_result(result, highlight=False):
	if type(result) == tuple:
		mean, std = result
		return f'${maybe_highlight(mean, highlight)}$ \\scriptsize{{\\raisebox{{1pt}}{{$\\pm {std}$}}}}'
	else:
		return f'${maybe_highlight(result, highlight)}$'

def format_row(buffer, environment, results):
	buffer_str = BUFFER_STRINGS[buffer]
	environment_str = ENVIRONMENT_STRINGS[environment]
	highlights = set_highlights(results)
	results_str = ' & '.join(format_result(result, h) for result, h in zip(results, highlights))
	row = f'{buffer_str} & {environment_str} & {results_str} \\\\ \n'
	return row

def format_buffer_block(algorithms, buffer, environments):
	block_str = '\\midrule\n'
	for environment in environments:
		results = [get_result(alg, buffer, environment) for alg in algorithms]
		row_str = format_row(buffer, environment, results)
		block_str += row_str
	return block_str

def format_algorithm(algorithm):
	algorithm_str = ALGORITHM_STRINGS.get(algorithm, algorithm)
	color = COLORS[algorithm] if algorithm in COLORS else 'black'
	return f'\multicolumn{{1}}{{r}}{{\\bf \\color{{{color}}} {algorithm_str}}}'

def format_algorithms(algorithms):
	return ' & '.join(format_algorithm(algorithm) for algorithm in algorithms)

def format_averages(means):
	prefix = f'\\multicolumn{{2}}{{c}}{{\\bf Average}} & '
	highlights = set_highlights(means)
	formatted = ' & '.join(maybe_highlight(str(mean), h) for mean, h in zip(means, highlights))
	return prefix + formatted

def format_averages_block(algorithms):
	means = [np.round(get_mean(MEANS[algorithm]), 1) for algorithm in algorithms]
	formatted = format_averages(means)

	formatted_block = (
		f'{formatted} \\hspace{{.6cm}} \\\\ \n'
	)
	return formatted_block

def format_color_defs():
	return '\n'.join([
			f'\\definecolor{{{key}}}{{HTML}}{{{val}}}'
			for key, val in COLOR_DEFS.items()
		]) + '\n'

def format_table(algorithms, buffers, environments):
	justify_str = 'll' + 'P' * (len(algorithms) - 1) + 'r'
	algorithm_str = format_algorithms(['Dataset', 'Environment'] + algorithms)
	averages_str = format_averages_block(algorithms)
	color_prefix = format_color_defs()
	table_prefix = (
		'\\newcolumntype{P}{>{\\raggedleft\\arraybackslash}X}\n'
		'\\begin{table*}[hb!]\n'
		'\\centering\n'
		'\\small\n'
		f'\\begin{{tabularx}}{{\\textwidth}}{{{justify_str}}}\n'
		'\\toprule\n'
		f'{algorithm_str} \\\\ \n'
	)
	table_suffix = (
		'\\midrule\n'
		f'{averages_str}'
		'\\bottomrule\n'
		'\\end{tabularx}\n'
		'\\vspace{-.0cm}\n'
		'\\caption{\n}\n'
		'\\label{table:locomotion}\n'
		'\\end{table*}'
	)
	blocks = ''.join(format_buffer_block(algorithms, buffer, environments) for buffer in buffers)
	table = (
		f'{color_prefix}'
		f'{table_prefix}'
		f'{blocks}'
		f'{table_suffix}'
	)
	return table


algorithms =['BC', 'CQL',  'IQL', 'DT', 'TT', 'MOPO', 'MOReL', 'MBOP', 'Diffuser']
buffers = ['medium-expert', 'medium', 'medium-replay']
environments = ['halfcheetah', 'hopper', 'walker2d']

table = format_table(algorithms, buffers, environments)
print(table)
