from scipy.optimize import minimize 
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dssmanipulation import dssToTree

def scipyoptimize(dssString):
	dssTree = dssToTree(dssString)

	# gets load loadshapes 
	load = [y.get('mult') for y in dssTree if y.get('object','').startswith('loadshape.6')]

	# combine loads into 1 list
	load = [load[i][1:len(load[i])-1].split(',') for i in range(len(load))]

	# flattens all load lists into one list for total load on the circuit 
	zipped_lists = zip(load[0],load[1], load[2], load[3], load[4], load[5], load[6],load[7], load[8], load[9], load[10], load[11],load[12], load[13], load[14])
	allLoads = [float(x) + float(y) + float(z) + float(a) + float(b) + float(c) + float(d) + float(e) + float(f) + float(g) + float(h) + float(i) + float(j) + float(k) + float(l) for (x,y,z,a,b,c,d,e,f,g,h,i,j,k,l) in zipped_lists]

	# gets generator loadshapes
	windGen = [y.get('mult') for y in dssTree if y.get('object','').startswith('loadshape.w')]
	windGen = windGen[0][1:len(windGen[0])-1].split(',')
	windGen = [float(i) for i in windGen]


	def objective(x):
	  return x 

	def constraint(x):
	  multipliedGeneration = [i * x for i in windGen]
	  newZip = zip(multipliedGeneration, allLoads)
	  difference = [i - j for (i,j) in newZip]
	  cumsum = np.cumsum(difference)
	  return min(cumsum)

	x0 = 10
	cons = {'type' : 'ineq', 'fun' : constraint}
	solution = minimize(objective,x0,method='SLSQP',constraints=cons)


	# redefine optimization function variables as global for graphing purposes 
	multipliedGeneration = [i * solution.x[0] for i in windGen]
	newZip = zip(multipliedGeneration, allLoads)
	difference = [i - j for (i,j) in newZip]
	cumsum = np.cumsum(difference)
	plotElements = pd.DataFrame({
	    'generation':multipliedGeneration,
	    'load':allLoads,
	    'storage':cumsum
	    })

	# plot generation, load, and storage
	feedInFigure = go.Figure(
	    [
	        go.Scatter(x=plotElements.generation.index, y=plotElements.generation, name='Wind Gen (kW)'),
	        go.Scatter(x=plotElements.load.index, y=plotElements.load, name='Load (kW)'),
	        go.Scatter(x=plotElements.storage.index, y=plotElements.storage, name='Storage (kW)'),
	    ],
	    go.Layout(
	        title = 'Combined Feed-in',
	        xaxis = {'title': 'Hour in 2019'},
	        yaxis = {'title': 'kW'},
	        legend = plotly_horiz_legend
	    )
	).show()
	print('Maximum storage capacity used by system in kW: ' + str(max(cumsum)))

	if __name__ == '__main__':
	fire.Fire()