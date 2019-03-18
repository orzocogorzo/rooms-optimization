from random import randint
import numpy as np

persons = ['Lucas','Tura','Claire','Pablo','Victor','Joni','Clara','Ruth','Guillem','Julia']
rooms = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV"]

def gen_matrix(x,y):
	return np.mat([[randint(1,10) for d in range(x)] for v in range(y)])

def ordinal_matrix(mat):
	order = [{col: mat[row, col] for col in range(mat.shape[1])} for row in range(mat.shape[0])]
	ordinal = list()
	for row in range(len(order)):
		row_order = sorted(order[row], key=lambda k: order[row][k])
		for i in range(len(row_order)):
			order[row][row_order[i]] = i+1
		
		ordinal.append([order[row][col] for col in range(mat.shape[1])])

	return np.mat(ordinal)

def sum_rows(mat):
	return mat.sum(axis=1)

def sum_cols(mat):
	return mat.sum(axis=0)

def avg_cols(mat):
	return [mat[:,i].mean() for i in range(mat.shape[1])]

def avg_rows(mat):
	return [v.mean() for v in mat]

def std_cols(mat):
	return [np.std(mat[:,i]) for i in range(mat.shape[1])]

def std_rows(mat):
	return [np.std(v) for v in mat] 

def dev_mat(mat, means):
	return mat - means

def drop_row(mat, rowIdx):
	if rowIdx < mat.shape[0]-1:
		return np.mat(mat[0:rowIdx].tolist() + mat[rowIdx+1:mat.shape[0]].tolist())
	else:
		return mat[0:rowIdx]

def drop_col(mat, colIdx):
	if colIdx < mat.shape[1]-1:
		return np.concatenate((mat[:,0:colIdx], mat[:,colIdx+1:mat.shape[1]]),axis=1)
	else:
		return mat[:,0:colIdx]

def optimize_cols(mat):
	deviations = dev_mat(mat, avg_cols(mat))
	return deviations - std_cols(mat)

def optimize_rows(mat):
	tmat = mat.transpose()
	deviations = dev_mat(tmat, avg_cols(tmat))
	return (deviations - std_cols(tmat)).transpose()

def gen_candidate(mat):
	col_candidates = optimize_cols(mat)
	row_candidates = optimize_rows(mat)
	candidates = (col_candidates + row_candidates)/2
	first = candidates.max()
	for row in range(mat.shape[0]):
		for col in range(mat.shape[1]):
			if candidates[row,col] == first:
				return [row,col]

def assign(mat, rows, cols):
		result = dict()
		rows = list(rows)
		cols = list(cols)
		while mat.shape[0] > 0:
			assignation = gen_candidate(mat)
			row = assignation[0]
			col = assignation[1]
			result[rows[row]] = cols[col]
			rows = rows[0:row] + rows[row+1:]
			cols = cols[0:col] + cols[col+1:]
			mat = drop_col(drop_row(mat, row), col)

		return result

def evaluate(result, mat, rows, cols):
	column_fit = 0
	row_fit = 0
	for k in result.keys():
		row = rows.index(k)
		col = cols.index(result[k])
		column_fit += mat[row,col]/mat[:,col].max()
		row_fit += mat[row,col]/mat[row,:].max()

	return {
		"column": column_fit/mat.shape[1],
		"row": row_fit/mat.shape[0],
		"global": (column_fit/mat.shape[1]+row_fit/mat.shape[0])/2
	}

