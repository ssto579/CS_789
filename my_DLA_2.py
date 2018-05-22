import numpy as np
import sys
import matplotlib.pylab as plt

import time


class the_grid():
	
	def __init__(self, x=5, y=5, num_iter=5, mode = 2, loop = 0, file = None):
		self.max_x = x
		self.max_y = y
		self.num_iter = num_iter
		self.loop = [loop %2, loop //2]
		self.posit = []
					 # is posit necessary? Not now, might end up being so later

		self.item_matrix = np.zeros((x,y))
		self.order_matrix = self.item_matrix * 0
		
		self.file = file
		
		
		self.out_print([self.max_x, self.max_y, self.num_iter, mode, loop], mode = 0)
		if mode == 0:
			self.posit.append([x//2, y//2])
			self.update_matrix(self.posit[-1], 3-1)
		elif mode == 1:
			self.item_matrix = np.random.rand(x,y)
			self.item_matrix = self.the_matrix < 0.001
			self.item_matrix= self.the_matrix.astype(int)
		else:
			self.item_matrix[-1,:] = 5
			self.item_matrix[-2,:] = 1
			
		
	def fix_pos(self, the_places):
		results = [0,0]
		the_maxes = [self.max_x, self.max_y]
		for i in range(2):
			if self.loop[i] == 0:
				results[i] = min(max(the_places[i],0), the_maxes[i]-1)
			else:
				results[i] = the_places[i] % the_maxes[i]
		return results

	def fix_adjacent_pos(self, the_places):
		[x,y] = the_places
		adj_places =  [[x,y-1],[x-1,y],[x,(y+1)],[x+1,y]]
		fixed_adj = []
		for c,d in adj_places:
			fixed_adj.append (self.fix_pos([c,d]))
			
		return fixed_adj
		
	def update_matrix(self, position, the_add):
		x,y = [int(i) for i in position ]
		self.item_matrix[x,y] += 4 # 
		self.order_matrix[x,y] = the_add + 0 # decide on k later
		for a,b in self.fix_adjacent_pos([x,y]):
			self.item_matrix[a,b] += 1
			self.order_matrix[a,b] = max(self.order_matrix[a,b], 1)
	


		
		# need to update this.
	def remove_dots(self):
		chance = 1/(2*len(self.posit)) # on average, one dot with 2 faces 
		chance_delete = []
		for [x,y] in self.posit:
			num_open = 0
			for a,b in self.fix_adjacent_pos([x,y]):
				if self.the_matrix[a,b] == 1:
					num_open += 1
			chance_delete.append(num_open*chance)
		deleted_items = []
		the_rand = np.random.random_sample(len(chance_delete))
		chance_delete = np.array(chance_delete)
		do_del = chance_delete > the_rand #booleans
		for i in range(len(chance_delete)-1, -1, -1):
			if do_del[i]:
				the_place = self.posit.pop(i)
				x,y = the_place
				self.posit.append( [str(x), str(y)])
				self.item_matrix[x, y ] -= 4 # not 0, since it's connected to something by necessity.
#				self.order_matrix[x, y ] -= 1 # not 0, since it's connected to something by necessity.
				for a,b in self.fix_adjacent_pos([x,y]):
					if a!=x or b!=y:
						self.item_matrix[a, b ] -= 1 
				
	
	def plot_grid(self):
		truth = self.order_matrix > 1
		truth = self.item_matrix > 4
		dot_only = self.order_matrix * truth # removes all the bordering ones
#		dot_only = self.item_matrix * truth # removes all the bordering ones
		dot_only = dot_only + truth *self.num_iter//6 # need to differentiate the early dots from the background of 0 for human eye. + max/6 seemed like the easiest way
		plt.matshow(dot_only, cmap=plt.cm.inferno)
		plt.colorbar()
		plt.show()
		
	def animated_grid(self):
		# based on https://stackoverflow.com/questions/11874767/real-time-plotting-in-while-loop-with-matplotlib
		t = .5
		black = '#000000'
		white = '#ffffff'
#		if (x,y) are  strings, idicates deletion. If numbers, indicates insertion
		plt.axis([0, self.max_y, 0, self.max_x]) # gives transpose to expected if x,y instead of y,x
		ax = plt.gca()
		ax.invert_yaxis()
		ax.set_facecolor(white)
		for [x,y] in self.posit:
			if [type(x) is int]:
				colour =   black
			else:
				colour =  white
			plt.scatter(y,x, c = colour)
			plt.pause(t)
			
		while True:
			plt.pause(t)
			
		
	def choose_one(self, selection, the_p):
		positions =list (np.random.choice(the_p.size, size = 1, p = the_p ))
		return positions
		
		

	
	def diffusion_version_2(self, selection = 0): # if selection is 0, select one item only, with prob matrix. Else, each item has a chance of being selected, with average chance being 1. If no items are selected, method 0 is used.
	# This is essentially max( poisson(selecton), 1)
		old_matrix = np.zeros((self.max_x,self.max_y))
		old_matrix = old_matrix + np.finfo(float).eps # fill the grid with a very small amount of items so as to prevent issues with tiny floating point numbers
		top_row_amount = float(1)
		old_matrix[0,:] = top_row_amount
		new_matrix = old_matrix*1
		kappa_divide = 100
		kappa = (float(top_row_amount)/kappa_divide)
		
		# I need to keep track of how many boxes each box is adjacent to. This works.
		border_matrix = old_matrix*0+4
		border_matrix = np.float_(border_matrix)
		border_matrix[0,:] -= 1
		border_matrix[-1,:] -= 1
		border_matrix[:,0] -= 1
		border_matrix[:,-1] -= 1

		position_func = self.choose_one
		iter_per_add = max(kappa_divide, self.max_x) * 3  # toy with later
		diffuse_into_filled_in = float(0.5) # also toy with later
		# I have diffusioon_into_filled_in to prevent the matrix being flooded with ones
		for i in range(self.num_iter):
			for j in range(iter_per_add):
				old_matrix = new_matrix*1
				old_matrix[0,:] = top_row_amount
				not_filled_in = self.item_matrix < 5
				old_matrix *= not_filled_in

				#roll right 1, remove leftmost row
				
				
				remove = (self.item_matrix*not_filled_in*diffuse_into_filled_in + (border_matrix-self.item_matrix*not_filled_in)*1)*old_matrix
				adj_matrices = [np.roll(old_matrix, 1, axis = 1) , np.roll(old_matrix, -1, axis = 1), np.roll(old_matrix, 1, axis = 0), np.roll(old_matrix, -1, axis = 0) ]
				adj_matrices[0][:,0] = 0
				adj_matrices[1][:,-1] = 0
				adj_matrices[2][0,:] = 0
				adj_matrices[3][-1,:] = 0
				add = new_matrix*0
				for matrix in adj_matrices:
					add = add + matrix
				delta_step = add- remove
				new_matrix = old_matrix + delta_step*kappa
			
			possibilites = new_matrix * not_filled_in * (self.item_matrix > 0)
#			print(possibilites)
#			print(type(possibilites))
#			print(type(possibilites[0][0]))
			the_p = np.reshape(possibilites, -1)
			the_p = the_p /np.sum(the_p)
			positions = []
			
#			if selection != 0:
#				positions = [i for i in range((possibilites.size)) if the_p[i]*selection >  np.random.random()]
#				self.selected.append(len(positions))
			
#			if selection == 0 or len(positions) == 0:
#				positions = [np.random.choice(possibilites.size, p = the_p )]
#				
#			positions =list (np.random.choice(possibilites.size, size = max(1, np.random.poisson(selection) ), replace=False, p = the_p ))
			positions = position_func(selection, the_p)
			for position in positions:
				[x,y] = [position// self.max_y , position% self.max_y ]
				self.update_matrix([x,y], i+3)
				self.posit.append([x, y])
				new_matrix[x,y] = 0
				if len(self.posit)% 100 == 0:
					print(len(self.posit))
			self.out_print(possibilites)
			self.out_print([x,y])






	def not_in_matrix(self, the_places):
		out = True
		for [a,b] in the_places:
			out = out and (self.item_matrix[a,b] < 1)
		return out #delete later
		
	def sum_adj(self, pos, a_matrix):
		the_sum = 0
		for [a,b] in self.fix_adjacent_pos(pos):
			if  self.not_in_matrix([ [a,b]]):
				the_sum += a_matrix[a,b]
		return the_sum
		
		
	def out_print(self, data, mode = 1):
		if self.file != None:
			if mode == 0: # more human readable IMO, loses data on large arrays due to how numpy works
				self.file.write(str(data))
				self.file.write('\n')
			else: #			less human readable IMO, doesn't lose data from numpy formatting.
				np.savetxt(self.file, data) 
				self.file.write('\n')
		else:
			print(data)

def main():
#	np.random.seed(seed = 123456789)
	the_time = int(time.time())
	print(the_time)
	np.random.seed(seed = the_time)
	out_file = open(('outputs/'+str(the_time)), 'w')
	out_file.write(str(the_time))
	out_file.write('\n')
	if len(sys.argv) == 1:
		#my_grid = the_grid(x=5, y=5, num_iter=5, mode = 2, loop = 0, file = out_file)
		#my_grid = the_grid(x=101, y=101, num_iter=1000, mode = 2, loop = 0, file = out_file)
		#my_grid = the_grid(x=31, y=11, num_iter=100, mode = 2, loop = 0, file = out_file)
		my_grid = the_grid(x=21, y=21, num_iter=20, mode = 2, loop = 0, file = out_file)
	else:
		states = [int(i) for i in sys.argv[1:] ]
		my_grid = the_grid(states[0], states[1], states[2], states [3], states[4], file = out_file)
	my_grid.diffusion_version_2()
	my_grid.plot_grid()
	my_grid.animated_grid()
	out_file.close()
main()