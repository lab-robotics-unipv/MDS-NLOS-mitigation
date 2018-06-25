from collections import defaultdict

import numpy as np
from sklearn.metrics import euclidean_distances


min_tag_id = 17          # specified in the nanoLoc user dev guide
line_not_error_code = 15 # default is 0
error_code_col = 3       # default is 2
anchor_col = 2           # default is 1


def parse_file(filename: str, anchors):
	data = []
	tag_data = defaultdict(lambda: [0] * len(anchors))
	new_tag = False
	with open(filename) as f:
		for line in f:
			if line.isspace() or not line.strip(): 
				if tag_data:
					yield build_dist_matrix(tag_data, anchors) # new chunk 
				continue
			tag_dist, anchor_id, tag_id, error_code = process_line(line)
			if error_code == line_not_error_code:
				tag_data[tag_id][anchor_id-1] = tag_dist

				
def build_dist_matrix(tag_data, anchors):
	distance_matrix = euclidean_distances(anchors)

	for tag_id in sorted(tag_data):
		pad = [0] * (distance_matrix.shape[1] - len(anchors))
		data = np.array(tag_data[tag_id] + pad)
		# append right-most distances for tag
		distance_matrix = np.hstack((distance_matrix, data.reshape(-1, 1)))
		# append lowest (symmetric with right-most) distances for tag
		distance_matrix = np.vstack((distance_matrix, np.append(data, 0).reshape(1, -1)))

	return distance_matrix


def process_line(line):
	tag_dist, tag_id, anchor_id, error_code = line[1:].strip().split(':')
	return  float(tag_dist), int(anchor_id), int(tag_id), int(error_code)



if __name__ == '__main__':
	from core.config import Config
	_ANCHORS = np.array([[0, 0], [3.25, 0], [-1.85, 4.00], [3.25, 4]])

	anchors = _ANCHORS
	for matrix in parse_file('data/3tags_20180523_17h19m51s_rawdata.txt', anchors):
		print(matrix)