# -*-coding:utf-8-*-
# @Time  : 2021/1/3 9:33
# @Author: sherlock
# @File  : plot_function.py
import numpy as np
from graphviz import Digraph


def plotmodel(B, latent, dag_name):
	# 画出以矩阵B为邻接矩阵的图
	dag = Digraph(dag_name, format='png')
	dag.attr('node', shape='circle')

	N = len(B)
	node_name_arr = []
	# 先画点
	for i in range(N):
		node_name = 'x' + str(i)
		node_name_arr.append(node_name)
		if i in latent:
			dag.node(node_name, node_name, style='dashed')
		else:
			dag.node(node_name, node_name)
	# 再画边
	for i in range(N):
		for j in range(N):
			if B[i, j] != 0.:
				if i in latent or j in latent:
					dag.edge(node_name_arr[j], node_name_arr[i], label=str(np.around(B[i, j], 4)), style='dashed')
				else:
					dag.edge(node_name_arr[j], node_name_arr[i], label=str(np.around(B[i, j], 4)))
	dag.view()
	# dag.render(dag_name)
