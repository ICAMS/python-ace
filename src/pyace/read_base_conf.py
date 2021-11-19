import numpy as np

class ReadPotInpt():
	def __init__(self, filename):
		self.rankmax = 5

		with open(filename) as f:
			self.lines = f.readlines()

		self.ranks = self.make_ranks()

	def make_ranks(self):
		config = self.lines[:13]
		base = self.lines[13:]

		r = self.get_params(config, self.rankmax)
		r_lines = self.r_data_split(base, r)
		rnks = [Rank(r_lines, i+1) for i in range(self.rankmax)]		

		return rnks

	def get_params(self, data, rmax=6):
		ranks = []
		for r in range(rmax):
			ranks.append(int(data[7+r].split()[1]))
	
		return ranks
	
	def r_data_split(self, data, r_size):
		r_data = []
		ind1 = 0
		for i,s in enumerate(r_size):
			mul = 2 if i < 3 else 3
			start = 1 if mul==2 else 2
	
			ind2 = ind1 + s*mul
			slc = [list(map(int, l.split())) for l in data[ind1:ind2][start::mul]]
			ind1 = ind2
			
			r_data.append(slc)
	
		return r_data

class Rank():
	def __init__(self, rank_data, rank):
		self.rank  = rank
		self._set_params(rank_data[rank-1])

	def _set_params(self, data):
		if self.rank == 1:
			self.ns = data
			self.ls = None
			self.LS = None
		elif self.rank == 2:
			self.ns = [d[:2] for d in data]
			self.ls = [d[2:]*2 for d in data]
			self.LS = None
		elif self.rank == 3:
			self.ns = [d[:3] for d in data]
			self.ls = [d[3:] for d in data]
			self.LS = [d[-1:] for d in data]
		elif self.rank == 4:
			self.ns = [d[:4] for d in data]
			self.ls = [d[4:-1] for d in data]
			self.LS = [d[-1:] for d in data]
		elif self.rank == 5:
			self.ns = [d[:5] for d in data]
			self.ls = [d[5:-2] for d in data]
			self.LS = [d[-2:] for d in data]

	def apply_constrains(self, nmax=2, lmax=0):
		ns = np.array(self.ns)
		cond = np.all(ns <= nmax, axis=1)# & (np.array(self.ls) <= lmax)
		cond2 = np.ones_like(cond)
		if self.ls is not None:
			ls = np.array(self.ls)
			cond2 = np.all(ls <= lmax, axis=1)
		mask = np.logical_and(cond, cond2)
		
		self.ns = list(map(list, ns[mask]))

		if self.ls is not None:
			self.ls = list(map(list, ls[mask]))
		if self.LS is not None:
			self.LS = list(map(list, np.array(self.LS)[mask]))

if __name__ == "__main__":
	name = 'bace.in'
	cfn = ReadPotInpt(name)
	#print(cfn.ranks[0].ns)
	cfn.ranks[1].apply_constrains(nmax=2, lmax=0)
	print(cfn.ranks[1].ls)
