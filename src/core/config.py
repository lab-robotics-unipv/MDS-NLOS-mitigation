import numpy as np

# number of dimensions of configuration 
M = 2 

# items for comparitive runs and plotting
ALGORITHMS = ('_smacof_single', '_smacof_with_anchors_single', '_smacof_with_distance_recovery_single')

# channel estimates for each anchor as seen by the tag
ALPHA = np.array([1.6, 2.4, 1.9, 1.51, 1.6, 2.0])
ALPHA = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

# define bounding box for node deployment
# larger values reduce the chances that nodes will overlap
X_MAX = 30
X_MIN = 0
Y_MAX = 20
Y_MIN = 0

_ORIGIN = np.array([X_MIN, Y_MIN])
_ANCHORS = np.array([_ORIGIN, 
			[X_MAX, Y_MIN], 
			[X_MAX, Y_MAX],
			[X_MIN, Y_MAX],
			[(X_MAX+X_MIN)/2.0, Y_MAX],
			[(X_MAX+X_MIN)/2.0, Y_MIN],
			[X_MIN, (Y_MAX+Y_MIN)/2.0],
			[X_MAX, (Y_MAX+Y_MIN)/2.0]])



class Config(object):
	'''parameter configuration for tag-anchor deployment'''
	
	def __init__(self, no_of_anchors=4, no_of_tags=10, noise=0, mu=0, **kwargs):
		self.no_of_anchors = no_of_anchors
		if no_of_tags < 1:
			raise ValueError('number of tags cannot be less that 1')
		self.no_of_tags = no_of_tags
		self.sigma = noise
		self.mu = mu
		for k, v in kwargs.items():
			setattr(self, k, v)

		self._points = self._anchors = self._tags = None

		# set homogenous weights 
		self.weights = np.ones((no_of_anchors + no_of_tags,)*2)
		np.fill_diagonal(self.weights, 0)


	def generate_points(self):
		#TODO: handle overlapping points
		if self._points is None or self._tags is None or self._anchors is None:            
			self._points = np.concatenate((self.anchors, self.tags))
		return self._points

	@property
	def points(self):
		return self.generate_points()

	@property
	def anchors(self):
		if self._anchors is None:
			if self.no_of_anchors < 1:
				self._anchors = np.array([[]])
			else:
				self._anchors = _ANCHORS[:self.no_of_anchors]
		return self._anchors

	@anchors.setter
	def anchors(self, value):
		self._anchors = value


	@property
	def tags(self):
		if self._tags is None:
			x0_displacements = np.random.choice(np.arange(1, X_MAX-X_MIN), size=self.no_of_tags)
			y0_displacements = np.random.choice(np.arange(1, Y_MAX-Y_MIN), size=self.no_of_tags)
			self._tags =  np.array([_ORIGIN + pt for pt in zip(x0_displacements, y0_displacements)])
			# self._tags =  np.array([[15, 10]for pt in zip(x0_displacements, y0_displacements)])
		return self._tags

	@property
	def no_of_tags(self):
		return self._no_of_tags

	@no_of_tags.setter
	def no_of_tags(self, val):
		self._tags = None
		self._no_of_tags = val

	@property
	def no_of_anchors(self):
		return self._no_of_anchors

	@no_of_anchors.setter
	def no_of_anchors(self, val):
		self._anchors = None
		self._no_of_anchors = val



# only use default for debugging
DEFAULT_POINTS = np.array([[0,20], [30,20], [30,0], [0,0], [18, 7], [4, 9], [10, 5], [7, 17], [15, 14], [12, 18], [22, 11]])
