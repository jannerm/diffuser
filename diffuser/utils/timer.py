import time

class Timer:

	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff