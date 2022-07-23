import os
import collections
import pickle

class Config(collections.Mapping):

	def __init__(self, model_class, savepath, verbose=True, **kwargs):
		self.model_class = model_class

		self._params = {}
		for key, val in kwargs.items():
			self._params[key] = val

		if verbose:
			print(self)

		if savepath is not None:
			fullpath = os.path.join(savepath, 'conf.pkl')
			pickle.dump(self, open(fullpath, 'wb'))
			print(f'Saved configuration to: {fullpath}\n')

	def __repr__(self):
		string = f'[ Config ]\n    class: {self.model_class}\n'
		for key in sorted(self._params.keys()):
			val = self._params[key]
			string += '    {}: {}\n'.format(key, val)
		return string

	def __iter__(self):
		return iter(self._params)

	def __getitem__(self, item):
		return self._params[item]

	def __len__(self):
		return len(self._params)

	def make(self):
		return self.model_class(**self._params)
