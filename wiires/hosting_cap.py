import networkx as nx
import os
import matplotlib.pyplot as plt
import fire
import pandas as pd
import opendssdirect as dss
import math
from wiires import dss_manipulation
import numpy as np
import multiprocessing
import tempfile
# import shutil
from functools import partial


# dss.Basic.DataPath("./data/")


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


def _getByName(tree, name):
    ''' Return first object with name in tree as an OrderedDict. '''
    matches =[]
    for x in tree:
        if x.get('object',''):
            if x.get('object','').split('.')[1] == name:
                matches.append(x)
    return matches[0]


def host_cap_plot(file_path, cap_dict, figsize=(20,20), output_path='./test', node_size=500, font_size=50):
	dssFileLoc = os.path.dirname(os.path.abspath(file_path))
	
	# Get bus coordinates 
	bus_coords = runDSS(file_path)
	bus_coords.columns = ['Bus', 'X', 'Y', 'radius']
	runDssCommand('Export voltages "' + dssFileLoc + '/volts.csv"')
	volts = pd.read_csv(dssFileLoc + '/volts.csv')
	G = nx.Graph()

	# Add bus nodes to graph and append bus coords to position dictionary 
	pos = {}
	for index, row in bus_coords.iterrows():
		try:
			bus_name = str(int(row['Bus']))
		except:
			bus_name = row['Bus']
		G.add_node(bus_name)
		pos[bus_name] = (float(row['X']), float(row['Y']))

	# Add bus labels to dictionary
	bus_labels = {}
	for index, row in volts.iterrows():
		bus_labels[row['Bus']] = row['Bus']

	# Read generation added from input dictionary 
	gen_added = {}
	for i in range(len(list(cap_dict.values()))):
		gen_added[list(cap_dict.keys())[i]] = list(cap_dict.values())[i]['gen_added']

	# Use dss_to_tree() to get load bus (671.1.2.3, 634.1, etc.) names 
	tree = dss_manipulation.dss_to_tree(file_path)
	load_buses = [y.get('bus1') for y in tree if y.get('object','').startswith('load.')]

	# Use dssToOmd() to get load names and coordinates 
	glm = dss_manipulation.dssToOmd(file_path, RADIUS=0.0002)
	glm_list = (list(glm.values()))
	load_labels = {}
	load_lat = []
	load_lon = []
	parents = []
	for i in glm_list:
		if i['object'] == 'load':
			load_labels[i['name']] = i['name']
			load_lat.append(i['latitude'])
			load_lon.append(i['longitude'])
			parents.append(i['parent'])
	labels = {**bus_labels,**load_labels}

	# Add node for every load 
	for i,j,k in zip(load_buses,load_lat,load_lon):
		G.add_node(i)
		pos[i] = (j,k)

	# Get the connecting edges using Pandas.
	edge_cm=[]
	lines = dss.utils.lines_to_dataframe()
	edges = []
	for index, row in lines.iterrows():
		#HACK: dss upercases everything in the coordinate output.
		bus1 = row['Bus1'].split('.')[0].upper()
		bus2 = row['Bus2'].split('.')[0].upper()
		edges.append((bus1, bus2))
		edge_cm.append('Black')
	# ADD EDGES BETWEEN LOADS AND BUSES
	for i,j in zip(parents,load_buses):
		edges.append((i,j))
		edge_cm.append('Gray')
	G.add_edges_from(edges)

	# Rescale from 0 to 1 in respect to generation added 
	rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
	gen_added_list = list(gen_added.values())
	gen_added_list_rescaled = rescale(gen_added_list)
	
	# Apply color coding to correct load nodes. Color buses gray
	node_cm = []
	cmap = plt.get_cmap()
	counter = -1
	for node in G:
		if node in load_buses:
			counter = counter + 1 
			rgba = cmap(gen_added_list_rescaled[counter])
			node_cm.append(convert_to_hex(rgba))
		else:
			node_cm.append('Gray')

	# Start drawing.
	plt.figure(figsize=figsize) 
	spring_pos = nx.drawing.layout.spring_layout(G)
	vmin = min(gen_added_list_rescaled)
	vmax = max(gen_added_list_rescaled)
	nx.draw_networkx(G, with_labels=True, node_color=node_cm, node_size=node_size, pos=spring_pos, edgelist=edges, edge_color=edge_cm, vmin=vmin, vmax=vmax)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	plt.colorbar(sm)
	plt.tight_layout()
	plt.savefig(dssFileLoc + '/' + output_path + '.png')
	plt.clf


def convert_to_hex(rgba_color) :
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#%02x%02x%02x' % (red, green, blue)


def host_cap_data(file_path, turb_min, turb_max, turb_kw, save_csv=False, output_path = './test', timeseries=False, 
	load_name=None, multiprocess=False, cores=8):
	tree = dss_manipulation.dss_to_tree(file_path)
	if load_name != None:
		load_buses = [y.get('bus1') for y in tree if load_name in y.get('object','')]
	else:
		# get names of all buses that have at least one load 
		load_buses = [y.get('bus1') for y in tree if y.get('object','').startswith('load.')]
		load_buses = list(dict.fromkeys(load_buses))
	# if snapshot capacity, arrange max load loadshapes and minimum generation loadshapes at front 
	if timeseries == False:	
		for y in tree:
			if y.get('object','').startswith('load.') and 'daily' in y.keys():
				tree = dss_manipulation.host_cap_dss_arrange(tree)
				break
	# adds generation at each load 15.6 kW at a time
	i  = None
	cap_dict = {}
	cap_list = []
	if multiprocess == True:
		pool = multiprocessing.Pool(processes=cores)
		func = partial(multiprocessor, turb_min, turb_max, tree, turb_kw, timeseries)
		print(f'Running multiprocessor {len(load_buses)} times with {cores} cores')
		cap_list.append(pool.map(func, load_buses))
		return cap_list
	elif multiprocess == False:
		for load in load_buses:
			print(load)
			for counter in range(turb_min, turb_max):
				dg_tree = dss_manipulation.add_turbine(tree, counter, load, turb_kw)
				dss_manipulation.tree_to_dss(dg_tree, './data/cap_circuit.dss')
				if timeseries == False:
					maximums, hour = newQstsPlot('./data/cap_circuit.dss', 60, 1)
				if timeseries == True:
					maximums, hour = newQstsPlot('./data/cap_circuit.dss', 60, 8760)
				print(counter, maximums, hour)
				if any(j >= 1.05 for j in maximums):
					cap_dict[load] = {'counter':counter,'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}
					break				
				else:
					cap_dict[load] = {'counter':'> ' + str(counter),'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}
					print("Load did not reach hosting capacity at " + str(counter + 1) + " " + str(turb_kw) + " kW turbines, or " + str(turb_kw * (counter + 1)) + " kW.")
		return cap_dict


def multiprocessor(turb_min, turb_max, tree, turb_kw, timeseries, load_buses):
	print("\n")
	print(load_buses)
	for counter in range(turb_min, turb_max):
		dg_tree = dss_manipulation.add_turbine(tree, counter, load_buses, turb_kw)
		# dss_manipulation.tree_to_dss(dg_tree, './data/cap_circuit.dss')
		# if timeseries == False:
		with tempfile.TemporaryDirectory() as path:
			os.chdir(path)
			print(os.getcwd())
			os.mkdir('./data')
			dss_manipulation.tree_to_dss(dg_tree, './data/cap_circuit.dss')
			if timeseries == False:
				maximums, hour = newQstsPlot('./data/cap_circuit.dss', 60, 1)
			if timeseries == True:
				maximums, hour = newQstsPlot('./data/cap_circuit.dss', 60, 8760)
			# except:
			# 	bug = open('./data/cap_circuit.dss')
			# 	fixer = open(f"./bug_files/{load_buses}.dss", "w")
			# 	fixer.write(bug.read())
			# 	bug.close()
			# 	fixer.close()
			#	return
		# if timeseries == True:
			# maximums, hour = newQstsPlot('./data/cap_circuit.dss', 60, 8760)
		print(counter, maximums, hour)
		if any(j >= 1.05 for j in maximums):
			# cap_dict[load_buses] = {'counter':counter,'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}
			print(f"Load reached hosting capacity at {counter + 1} {turb_kw} kW turbines, or {turb_kw * (counter + 1)} kW.")
			# return (counter, turb_kw, turb_kw*counter, hour, maximums)
			return {'load':load_buses,'counter':counter,'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}				
	# cap_dict[load_buses] = {'counter':'> ' + str(counter),'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}
	# print("Load did not reach hosting capacity at " + str(counter + 1) + " " + str(turb_kw) + " kW turbines, or " + str(turb_kw * (counter + 1)) + " kW.")
	print(f"Load did not reach hosting capacity at {counter + 1} {turb_kw} kW turbines, or {turb_kw * (counter + 1)} kW.")
	# return (counter, turb_kw, turb_kw*counter, hour, maximums)
	return {'load':load_buses,'counter':'> ' + str(counter),'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}


def get_host_cap(file_path, turb_min, turb_max, turb_kw, save_csv=False, timeseries=False, load_name=None, figsize=(20,20), 
	output_path='./test', node_size=500, font_size=50, multiprocess=False, cores=2):
	cap_dict = host_cap_data(file_path, turb_min, turb_max, turb_kw, save_csv, output_path, timeseries, load_name, multiprocess, cores)
	# if type(cap_dict) is dict: 
		# print("cap_dict", cap_dict)
	if type(cap_dict) is list:
		# print("cap list", cap_dict)
		cap_list = cap_dict[0]
		cap_dict = {}
		for item in cap_list:
			cap_dict[item['load']] = {
			'counter':item['counter'],
			'turb_kw':item['turb_kw'],
			'gen_added':item['gen_added'],
			'hour':item['hour'],
			'maximums':item['maximums']
			}
	if save_csv==True:
		cap_df = pd.DataFrame()
		cap_df = cap_df.from_dict(cap_dict, orient='columns', dtype=None, columns=None)
		print(cap_df)
		print(os.getcwd())
		cap_df.to_csv(f'./data/{output_path}.csv')
	host_cap_plot(file_path, cap_dict, figsize, output_path, node_size, font_size)


def newQstsPlot(filePath, stepSizeInMinutes, numberOfSteps, keepAllFiles=False, actions={}):
	''' QSTS with native opendsscmd binary to avoid segfaults in opendssdirect. '''
	dssFileLoc = os.path.dirname(os.path.abspath(filePath))
	dss_run_file = ''
	# volt_coord = runDSS(filePath)
	dss_run_file += f'redirect {dssFileLoc}/cap_circuit.dss\n'
	dss_run_file += f'set datapath="{dssFileLoc}"\n'
	dss_run_file += f'calcvoltagebases\n'
	# Attach Monitors
	tree = dss_manipulation.dss_to_tree(filePath)
	mon_names = []
	circ_name = 'NONE'
	base_kvs = pd.DataFrame()
	for ob in tree:
		obData = ob.get('object','NONE.NONE')
		obType, name = obData.split('.', 1)
		mon_name = f'mon{obType}-{name}'
		if obData.startswith('circuit.'):
			circ_name = name
		elif ob.get('object','').startswith('load.'):
			dss_run_file += f'new object=monitor.{mon_name} element={obType}.{name} terminal=1 mode=0\n'
			mon_names.append(mon_name)
			new_kv = pd.DataFrame({'kv':[float(ob.get('kv',1.0))],'Name':['monload-' + name]})
			base_kvs = base_kvs.append(new_kv)
	# Run DSS
	dss_run_file += f'set mode=yearly stepsize={stepSizeInMinutes}m \n'
	if actions == {}:
		# Run all steps directly.
		dss_run_file += f'set number={numberOfSteps}\n'
		dss_run_file += 'solve\n'
	else:
		# Actions defined, run them at the appropriate timestep.
		dss_run_file += f'set number=1\n'
		for step in range(1, numberOfSteps+1):
			action = actions.get(step)
			if action != None:
				print(f'Step {step} executing:', action)
				dss_run_file += action
			dss_run_file += 'solve\n'
	# Export all monitors
	for name in mon_names:
		dss_run_file += f'export monitors monitorname={name}\n'
	# Write runner file and run.
	with open(f'{dssFileLoc}/dss_run_file.dss', 'w') as run_file:
		run_file.write(dss_run_file)
	os.system(f'opendsscmd {dssFileLoc}/dss_run_file.dss')

	# Aggregate monitors
	all_load_df = pd.DataFrame()
	for name in mon_names:
		csv_path = f'{dssFileLoc}/{circ_name}_Mon_{name}.csv'
		df = pd.read_csv(f'data/{circ_name}_Mon_{name}.csv')
		if name.startswith('monload-'):
			# # TODO: TEST THAT the commented out phasing code below works after new_newQSTSplot() is updated
			# # reassign V1 single phase voltages outputted by DSS to the appropriate column and filling Nans for neutral phases (V2)
			# # three phase print out should work fine as is
			ob_name = name.split('-')[1]
			# # print("ob_name:", ob_name)
			the_object = _getByName(tree, ob_name)
			# # print("the_object:", the_object)
			# # create phase list, removing neutral phases
			phase_ids = the_object.get('bus1','').replace('.0','').split('.')[1:]
			# # print("phase_ids:", phase_ids)
			# # print("headings list:", df.columns)
			if phase_ids == ['1']:
				df[[' V2']] = np.NaN
				df[[' V3']] = np.NaN
			elif phase_ids == ['2']:
				df[[' V2']] = df[[' V1']]
				df[[' V1']] = np.NaN
				df[[' V3']] = np.NaN
			elif phase_ids == ['3']:
				df[[' V3']] = df[[' V1']]
				df[[' V1']] = np.NaN
				df[[' V2']] = np.NaN
			# # print("df after phase reassignment:")
			# # print(df.head(10))
			df['Name'] = name
			all_load_df = pd.concat([all_load_df, df], ignore_index=True, sort=False)
			#pd.set_option('display.max_columns', None)
			#print("all_load_df:", df.head(50))
		if not keepAllFiles:
			os.remove(csv_path)

	# Collect switching actions
	for key, ob in actions.items():
		if ob.startswith('open'):
			switch_ob = ob.split()
			ob_name = switch_ob[1][7:]
			new_row = {'hour':key, 't(sec)':0.0,'Tap(pu)':1,'Type':'Switch','Name':ob_name}
			all_control_df = all_control_df.append(new_row, ignore_index=True)
	for key, ob in actions.items():
		if ob.startswith('close'):
			switch_ob = ob.split()
			ob_name = switch_ob[1][7:]
			new_row = {'hour':key, 't(sec)':0.0,'Tap(pu)':1,'Type':'Switch','Name':ob_name}
			all_control_df = all_control_df.append(new_row, ignore_index=True)
	# Write final aggregates
	if not all_load_df.empty:
		all_load_df.sort_values(['Name','hour'], inplace=True)
		all_load_df.columns = all_load_df.columns.str.replace(r'[ "]','',regex=True)
		all_load_df = all_load_df.join(base_kvs.set_index('Name'), on='Name')
		# TODO: insert ANSI bands here based on base_kv?  How to not display two bands per load with the appended CSV format?
		all_load_df['V1(PU)'] = all_load_df['V1'].astype(float) / (all_load_df['kv'].astype(float) * 1000.0)
		# HACK: reassigning 0V to "NaN" as below does not removes 0V phases but could impact 2 phase systems
		#all_load_df['V2'][(all_load_df['VAngle2']==0) & (all_load_df['V2']==0)] = "NaN"
		all_load_df['V2(PU)'] = all_load_df['V2'].astype(float) / (all_load_df['kv'].astype(float) * 1000.0)
		all_load_df['V3(PU)'] = all_load_df['V3'].astype(float) / (all_load_df['kv'].astype(float) * 1000.0)
		all_load_df.to_csv(f'{dssFileLoc}/timeseries_load.csv', index=False)
		PU1 = all_load_df['V1(PU)']
		PU2 = all_load_df['V2(PU)']
		PU3 = all_load_df['V3(PU)']
		maximums = all_load_df[['V1(PU)','V2(PU)','V3(PU)']].max()
		max_v1 = all_load_df['V1(PU)'].max()
		index1 = PU1[PU1 == maximums[0]].index[0]
		hour1 = all_load_df.loc[index1, 'hour']
		
		max_v2 = all_load_df['V2(PU)'].max()
		index2 = PU2[PU2 == maximums[1]].index[0]
		hour2 = all_load_df.loc[index2, 'hour']

		max_v3 = all_load_df['V3(PU)'].max()
		index3 = PU3[PU3 == maximums[2]].index[0]
		hour3 = all_load_df.loc[index3, 'hour']

		hours = hour1, hour2, hour3

		maximums_list = maximums.tolist()
		return maximums_list, hours[maximums_list.index(max(maximums_list))]


if __name__ == "__main__":
	get_host_cap('./data/lehigh.dss', 1, 5, 10_000, save_csv=False, timeseries=False, load_name=None, figsize=(20,20), 
		output_path='plot_labels_test', node_size=500, font_size=50, multiprocess=True, cores=2)


# if __name__ == '__main__':
	# fire.Fire()
