from scipy.optimize import minimize 
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dss_manipulation import dss_to_tree
import fire


def scipy_optimize(dss_tree):
	# gets load loadshapes 
	load = [y.get('mult') for y in dss_tree if y.get('object','').startswith('loadshape.6')]
	# combine loads into 1 list
	load = [load[i][1:len(load[i])-1].split(',') for i in range(len(load))]
	# flattens all load lists into one list for total load on the circuit 
	zipped_lists = zip(load[0],load[1], load[2], load[3], load[4], load[5], load[6],load[7], load[8], load[9], load[10], load[11],load[12], load[13], load[14])
	all_loads = [float(x) + float(y) + float(z) + float(a) + float(b) + float(c) + float(d) + float(e) + float(f) + float(g) + float(h) + float(i) + float(j) + float(k) + float(l) for (x,y,z,a,b,c,d,e,f,g,h,i,j,k,l) in zipped_lists]
	# gets generator loadshapes
	wind_gen = [y.get('mult') for y in dss_tree if y.get('object','').startswith('loadshape.w')]
	wind_gen = wind_gen[0][1:len(wind_gen[0])-1].split(',')
	wind_gen = [float(i) for i in wind_gen]

	def objective(x):
	  return x 

	def constraint(x):
	  multiplied_generation = [i * x for i in wind_gen]
	  new_zip = zip(multiplied_generation, all_loads)
	  difference = [i - j for (i,j) in new_zip]
	  cumsum = np.cumsum(difference)
	  return min(cumsum)

	x0 = 10
	cons = {'type' : 'ineq', 'fun' : constraint}
	solution = minimize(objective,x0,method='SLSQP',constraints=cons)
	# redefine optimization function variables as global for graphing purposes 
	multiplied_generation = [i * solution.x[0] for i in wind_gen]
	new_zip = zip(multiplied_generation, all_loads)
	difference = [i - j for (i,j) in new_zip]
	cumsum = np.cumsum(difference)
	plot_elements = pd.DataFrame({
	    'generation':multiplied_generation,
	    'load':all_loads,
	    'storage':cumsum
	    })
	# plot generation, load, and storage
	plotly_horiz_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
	feed_in_figure = go.Figure(
	    [
	        go.Scatter(x=plot_elements.generation.index, y=plot_elements.generation, name='Wind Gen (kW)'),
	        go.Scatter(x=plot_elements.load.index, y=plot_elements.load, name='Load (kW)'),
	        go.Scatter(x=plot_elements.storage.index, y=plot_elements.storage, name='Storage (kW)'),
	    ],
	    go.Layout(
	        title = 'Combined Feed-in',
	        xaxis = {'title': 'Hour in 2019'},
	        yaxis = {'title': 'kW'},
	        legend = plotly_horiz_legend
	    )
	).show()
	print('Maximum storage capacity used by system in kW: ' + str(max(cumsum)))


def scipy_optimize_dss(file_path):
	dss_tree = dss_to_tree(file_path)
	scipy_optimize(dss_tree)


if __name__ == '__main__':
	fire.Fire()