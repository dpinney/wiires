import networkx as nx
import os
import matplotlib.pyplot as plt
import fire
import pandas as pd
import opendssdirect as dss
import math


dss.Basic.DataPath("./data/")


def runDssCommand(dsscmd):
	from opendssdirect import run_command, Error
	x = run_command(dsscmd)
	latest_error = Error.Description()
	if latest_error != '':
		print('OpenDSS Error:', latest_error)
	return x


def runDSS(dssFilePath, keep_output=True):
	''' Run DSS circuit definition file and set export path. Generates file named coords.csv in directory of input file.
	To avoid generating this file, set the 'keep_output' parameter to False.'''
	# Check for valid .dss file
	assert '.dss' in dssFilePath.lower(), 'The input file must be an OpenDSS circuit definition file.'
	# Get paths because openDss doesn't like relative paths.
	fullPath = os.path.abspath(dssFilePath)
	dssFileLoc = os.path.dirname(fullPath)
	try:
		with open(fullPath):
			pass
	except Exception as ex:
		print('While accessing the file located at %s, the following exception occured: %s'%(dssFileLoc, ex))
	runDssCommand('Clear')
	runDssCommand('Redirect "' + fullPath + '"')
	runDssCommand('Solve')
	# also generate coordinates.
	# TODO?: Get the coords as a separate function (i.e. getCoords() below) and instead return dssFileLoc.
	x = runDssCommand('Export BusCoords "' + dssFileLoc + '/coords.csv"')
	coords = pd.read_csv(dssFileLoc + '/coords.csv', dtype=str, header=None, names=['Element', 'X', 'Y'])
	# TODO: reverse keep_output logic - Should default to cleanliness. Requires addition of 'keep_output=True' to all function calls.
	if not keep_output:
		os.remove(x)
	hyp = []
	for index, row in coords.iterrows():
		hyp.append(math.sqrt(float(row['X'])**2 + float(row['Y'])**2))
	coords['radius'] = hyp
	return coords


def networkPlot(filePath, figsize=(20,20), output_name='networkPlot.png', show_labels=True, node_size=300, font_size=8):
	''' Plot the physical topology of the circuit. '''
	dssFileLoc = os.path.dirname(os.path.abspath(filePath))
	coords = runDSS(filePath)
	runDssCommand('Export voltages "' + dssFileLoc + '/volts.csv"')
	volts = pd.read_csv(dssFileLoc + '/volts.csv')
	coords.columns = ['Bus', 'X', 'Y', 'radius']
	G = nx.Graph()
	# Get the coordinates.
	pos = {}
	for index, row in coords.iterrows():
		try:
			bus_name = str(int(row['Bus']))
		except:
			bus_name = row['Bus']
		G.add_node(bus_name)
		pos[bus_name] = (float(row['X']), float(row['Y']))
	# Get the connecting edges using Pandas.
	lines = dss.utils.lines_to_dataframe()
	edges = []
	for index, row in lines.iterrows():
		#HACK: dss upercases everything in the coordinate output.
		bus1 = row['Bus1'].split('.')[0].upper()
		bus2 = row['Bus2'].split('.')[0].upper()
		edges.append((bus1, bus2))
	G.add_edges_from(edges)
	# We'll color the nodes according to voltage.
	volt_values = {}
	labels = {}
	for index, row in volts.iterrows():
		volt_values[row['Bus']] = row[' pu1']
		labels[row['Bus']] = row['Bus']
	colorCode = [volt_values.get(node, 0.0) for node in G.nodes()]


	# find highest voltage, increase to 1.05, increase all voltages by same amount, output amount 
	all_values = volt_values.values()
	max_volt = max(all_values)
	big_bus = max(volt_values, key=volt_values.get)
	if max_volt < 1.05:
		diff = 1.05 - max_volt
		volt_values = {key: val + diff for key, val in volt_values.items()}
		print("Reached hosting capacity at " + big_bus)
		print(big_bus + " was " + str(max_volt) + ". It was " + str(diff) + " under 1.05")
	else: 
		print("Circuit is already at hosting capacity at " + big_bus)
		print(big_bus + " is at " + str(max_volt))

	# Start drawing.
	plt.figure(figsize=figsize) 
	nodes = nx.draw_networkx_nodes(G, pos, node_color=colorCode, node_size=node_size)
	edges = nx.draw_networkx_edges(G, pos)
	if show_labels:
		nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
	plt.colorbar(nodes)
	plt.title('Network Voltage Layout')
	plt.tight_layout()
	plt.savefig(dssFileLoc + '/' + output_name)
	plt.clf


if __name__ == '__main__':
	fire.Fire()


networkPlot("mod_lehigh.dss")