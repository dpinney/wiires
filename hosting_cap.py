import networkx as nx
import os
import matplotlib.pyplot as plt
import fire
import pandas as pd
import opendssdirect as dss
import math
import dss_manipulation
import numpy as np


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


def network_plot(file_path, cap_dict, hour_input, figsize=(20,20), output_name='network_plot.png', show_labels=True, node_size=500, font_size=50):
	dssFileLoc = os.path.dirname(os.path.abspath(file_path))
	
	# BUSES
	bus_coords = runDSS(file_path)
	bus_coords.columns = ['Bus', 'X', 'Y', 'radius']
	runDssCommand('Export voltages "' + dssFileLoc + '/volts.csv"')
	volts = pd.read_csv(dssFileLoc + '/volts.csv')
	G = nx.Graph()
	# add bus nodes to graph and append bus coords to position dictionary 
	pos = {}
	for index, row in bus_coords.iterrows():
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

	# add bus nodes to dictionary for later coloring according to voltage
	# bus_volts = {}
	bus_labels = {}
	# volts['pu max'] = volts[[' pu1',' pu2',' pu3']].max(axis=1)
	for index, row in volts.iterrows():
	# 	bus_volts[row['Bus']] = row['pu max'] 
		bus_labels[row['Bus']] = row['Bus']

	# LOADS
	# get every hour_input load name and maximum PU voltage and create new volts and labels objects 
	# timeseries_load = pd.read_csv('timeseries_load.csv')
	# timeseries_load = timeseries_load.loc[timeseries_load.hour==hour_input]
	# timeseries_load['PU max'] = timeseries_load.loc[:,'V1(PU)':'V3(PU)'].max(axis=1)
	# load_volts = {}
	# load_labels = {}
	# for index, row in timeseries_load.iterrows():
	# 	load_volts[row['Name']] = row['PU max']
	# 	load_labels[row['Name']] = row['Name']

	loadbus_names = {}	
	counters = {}
	hours = {}
	maximums = {}
	gen_added = {}
	for i in range(len(list(cap_dict.values()))):
		loadbus_names[list(cap_dict.keys())[i]] = list(cap_dict.keys())[i]
		counters[list(cap_dict.keys())[i]] = list(cap_dict.values())[i]['counter']
		hours[list(cap_dict.keys())[i]] = list(cap_dict.values())[i]['hour']
		maximums[list(cap_dict.keys())[i]] = list(cap_dict.values())[i]['maximums']
		gen_added[list(cap_dict.keys())[i]] = list(cap_dict.values())[i]['gen_added']
	# print(loadbus_names)
	# print(counters)
	# print(hours)
	# print(maximums)
	# print(gen_added)

	# use dss_to_tree() to get bus names 
	tree = dss_manipulation.dss_to_tree(file_path)
	load_buses = [y.get('bus1') for y in tree if y.get('object','').startswith('load.')]

	# use dssToOmd() to get load names and coordinates 
	glm = dss_manipulation.dssToOmd(file_path, RADIUS=0.0002)
	glm_list = (list(glm.values()))
	load_labels = {}
	load_lat = []
	load_lon = []
	for i in glm_list:
		if i['object'] == 'load':
			load_labels[i['name']] = i['name']
			load_lat.append(i['latitude'])
			load_lon.append(i['longitude'])
	labels = {**bus_labels,**load_labels}

	# add node for every load 
	for i,j,k in zip(load_buses,load_lat,load_lon):
		G.add_node(i)
		pos[i] = (j,k)

	# volt_values = {**bus_volts,**gen_added}

	# big_load = max(gen_added, key=gen_added.get)
	colorCode = [gen_added.get(node, 0.0) for node in G.nodes()]
	# max_volt = max(colorCode)

	# set edge color to red if node hit hosting capacity 
	# edge_colors = []
	# for node in G.nodes():
	# 	if gen_added.get(node, 0.0) >= 1.05:
	# 		edge_colors.append("red")
	# 	else:
	# 		edge_colors.append("black")	

	# Start drawing.
	plt.figure(figsize=figsize) 
	# nodes = nx.draw_networkx_nodes(G, pos, node_color=colorCode, node_size=node_size, edgecolors=edge_colors)
	# edges = nx.draw_networkx_edges(G, pos)
	# if show_labels:
	# 	nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
	# plt.colorbar(nodes)
	spring_pos = nx.drawing.layout.spring_layout(G)
	nx.draw_networkx(G, with_labels=True, node_color=colorCode, node_size=node_size, pos=spring_pos)
	# if hour_input != None:
		# plt.title("Circuit reached hosting capacity at " + str(counter + 1) + " 15.6 kW turbines, or " + str(15.6 * (counter + 1)) + " kW of distributed generation per load. Node " + big_load + " reached hosting capacity at a per unit voltage of " + str(max_volt) + " in hour " + str(hour_input) + ".")
	# else:
		# plt.title("Circuit reached hosting capacity at " + str(counter + 1) + " 15.6 kW turbines, or " + str(15.6 * (counter + 1)) + " kW of distributed generation per load. Node " + big_load + " reached hosting capacity at a per unit voltage of " + str(max_volt) + ".")
	plt.tight_layout()
	plt.savefig(dssFileLoc + '/' + output_name)
	# plt.show()
	plt.clf


def get_hosting_cap(file_path, turb_min, turb_max, turb_kw, snapshot_or_timeseries='snapshot', load_name='',):
	tree = dss_manipulation.dss_to_tree(file_path)
	if load_name != '':
		load_buses = [y.get('bus1') for y in tree if load_name in y.get('object','')]
	else:
		# get names of all buses that have at least one load 
		load_buses = [y.get('bus1') for y in tree if y.get('object','').startswith('load.')]
		load_buses = list(dict.fromkeys(load_buses))
	# arrange max load loadshapes and minimum generation loadshapes at front 
	if snapshot_or_timeseries == 'snapshot':	
		for y in tree:
			if y.get('object','').startswith('load.') and 'daily' in y.keys():
				tree = dss_manipulation.host_cap_dss_arrange(tree, None)
				break
	# adds generation at each load 15.6 kW at a time
	i  = None
	cap_dict = {}
	for load in load_buses:
		print(load)
		for counter in range(turb_min, turb_max):
			dg_tree = dss_manipulation.add_turbine(tree, counter, load, turb_kw)
			dss_manipulation.tree_to_dss(dg_tree, 'cap_circuit.dss')
			maximums, hour = newQstsPlot('cap_circuit.dss', 60, 8760)
			print(counter, maximums, hour)
			if any(j >= 1.05 for j in maximums):
				cap_dict[load] = {'counter':counter,'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}
				break				
		else:
			cap_dict[load] = {'counter':'> ' + str(counter),'turb_kw':turb_kw,'gen_added':(turb_kw*counter),'hour':hour,'maximums':maximums}
			print("Load did not reach hosting capacity at " + str(counter + 1) + " " + str(turb_kw) + " kW turbines, or " + str(turb_kw * (counter + 1)) + " kW.")
	if snapshot_or_timeseries == 'timeseries':
		ts_tree = dss_manipulation.dss_to_tree('cap_circuit.dss')
		ts_tree = dss_manipulation.host_cap_dss_arrange(ts_tree, hour)
		dss_manipulation.tree_to_dss(ts_tree, 'cap_circuit.dss')
	network_plot('cap_circuit.dss', cap_dict, 1, figsize=(20,20), output_name='network_plot.png')
	print(cap_dict)
	return cap_dict


def newQstsPlot(filePath, stepSizeInMinutes, numberOfSteps, keepAllFiles=False, actions={}):
	''' Use monitor objects to generate voltage values for a timeseries powerflow. '''
	dssFileLoc = os.path.dirname(os.path.abspath(filePath))
	volt_coord = runDSS(filePath)
	runDssCommand(f'set datapath="{dssFileLoc}"')
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
			runDssCommand(f'new object=monitor.{mon_name} element={obType}.{name} terminal=1 mode=0')
			mon_names.append(mon_name)
			new_kv = pd.DataFrame({'kv':[float(ob.get('kv',1.0))],'Name':[name]})
			base_kvs = base_kvs.append(new_kv)
	# Run DSS
	runDssCommand(f'set mode=yearly stepsize={stepSizeInMinutes}m ')
	if actions == {}:
		# Run all steps directly.
		runDssCommand(f'set number={numberOfSteps}')
		runDssCommand('solve')
	else:
		# Actions defined, run them at the appropriate timestep.
		runDssCommand(f'set number=1')
		for step in range(1, numberOfSteps+1):
			action = actions.get(step)
			if action != None:
				print(f'Step {step} executing:', action)
				runDssCommand(action)
			runDssCommand('solve')
	# Export all monitors
	for name in mon_names:
		runDssCommand(f'export monitors monitorname={name}')
	# Aggregate monitors
	all_load_df = pd.DataFrame()
	for name in mon_names:
		csv_path = f'{dssFileLoc}/{circ_name}_Mon_{name}.csv'
		df = pd.read_csv(f'{circ_name}_Mon_{name}.csv')
		if name.startswith('monload-'):
			# reassign V1 single phase voltages outputted by DSS to the appropriate column and filling Nans for neutral phases (V2)
			# three phase print out should work fine as is
			ob_name = name.split('-')[1]
			the_object = _getByName(tree, ob_name)
			# print("the_object:", the_object)
			# create phase list, removing neutral phases
			phase_ids = the_object.get('bus1','').replace('.0','').split('.')[1:]
			# print("phase_ids:", phase_ids)
			# print("headings list:", df.columns)
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
			# print("df after phase reassignment:")
			# print(df.head(10))
			df['Name'] = ob_name
			all_load_df = pd.concat([all_load_df, df], ignore_index=True, sort=False)
			# # pd.set_option('display.max_columns', None)
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
	# Write final aggregate
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


def _getByName(tree, name):
    ''' Return first object with name in tree as an OrderedDict. '''
    matches =[]
    for x in tree:
        if x.get('object',''):
            if x.get('object','').split('.')[1] == name:
                matches.append(x)
    return matches[0]


if __name__ == '__main__':
	fire.Fire()


cap_dict = get_hosting_cap('lehigh.dss', 1, 101, 100_000, 'timeseries')
# cap_dict = {'671.1.2.3': {'counter': 1, 'turb_kw': 100000, 'gen_added': 100000, 'hour': 1, 'maximums': [1.4442708333333334, 1.3788666666666667, 1.4306541666666668]}, '634.1': {'counter': 10, 'turb_kw': 100000, 'gen_added': 1000000, 'hour': 25, 'maximums': [1.0571407942238267, 1.0131666666666665, 1.0176083333333334]}, '645.2': {'counter': '> 100', 'turb_kw': 100000, 'gen_added': 10000000, 'hour': 4, 'maximums': [0.9734791666666667, 0.9802333333333333, 0.9837416666666666]}, '646.2': {'counter': '> 100', 'turb_kw': 100000, 'gen_added': 10000000, 'hour': 4, 'maximums': [0.9734791666666667, 0.9802333333333333, 0.9837416666666666]}, '692.1.2.3': {'counter': 1, 'turb_kw': 100000, 'gen_added': 100000, 'hour': 1, 'maximums': [1.4442708333333334, 1.3788666666666667, 1.4306583333333334]}, '675.1.2.3': {'counter': 1, 'turb_kw': 100000, 'gen_added': 100000, 'hour': 1, 'maximums': [1.4962208333333333, 1.4263041666666667, 1.46565]}, '611.3': {'counter': '> 100', 'turb_kw': 100000, 'gen_added': 10000000, 'hour': 4, 'maximums': [0.9734791666666667, 0.9802333333333333, 0.9837416666666666]}, '652.3': {'counter': '> 100', 'turb_kw': 100000, 'gen_added': 10000000, 'hour': 4, 'maximums': [0.9734791666666667, 0.9802333333333333, 0.9837416666666666]}, '670.1': {'counter': 1, 'turb_kw': 100000, 'gen_added': 100000, 'hour': 4, 'maximums': [1.4016208333333333, 1.4044583333333334, 1.4172083333333334]}, '670.2': {'counter': 1, 'turb_kw': 100000, 'gen_added': 100000, 'hour': 4, 'maximums': [1.4016208333333333, 1.4044583333333334, 1.4172083333333334]}, '670.3': {'counter': 1, 'turb_kw': 100000, 'gen_added': 100000, 'hour': 4, 'maximums': [1.4016208333333333, 1.4044583333333334, 1.4172083333333334]}}
cap_df = pd.DataFrame()
cap_df = cap_df.from_dict(cap_dict, orient='columns', dtype=None, columns=None)
cap_df.to_csv('cap_df.csv')
# network_plot('cap_circuit.dss', cap_dict, 1, figsize=(20,20), output_name='network_plot.png', show_labels=True, node_size=500, font_size=25)
