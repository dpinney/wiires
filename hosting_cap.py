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


def network_plot(file_path, i, figsize=(50,50), output_name='networkPlot.png', show_labels=True, node_size=300, font_size=10):
	''' Plot the physical topology of the circuit. '''
	dssFileLoc = os.path.dirname(os.path.abspath(file_path))
	coords = runDSS(file_path)
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
	all_values = volt_values.values()
	max_volt = max(all_values)
	big_bus = max(volt_values, key=volt_values.get)
	colorCode = [volt_values.get(node, 0.0) for node in G.nodes()]

	# set edge color to red if node hit hosting capacity 
	edge_colors = []
	for node in G.nodes():
		if volt_values.get(node, 0.0) >= 1.05:
			edge_colors.append("red")
		else:
			edge_colors.append("black")

	# Start drawing.
	plt.figure(figsize=figsize) 
	nodes = nx.draw_networkx_nodes(G, pos, node_color=colorCode, node_size=node_size, edgecolors=edge_colors)
	edges = nx.draw_networkx_edges(G, pos)
	if show_labels:
		nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
	plt.colorbar(nodes)
	# plt.title('Network Voltage Layout')
	plt.title("Circuit reached hosting capacity at " + str(i + 1) + " 15.6 kW turbines, or " + str(15.6 * (i + 1)) + " kW of distributed generation per load. Node " + big_bus + " reached hosting capacity at " + str(max_volt))
	plt.tight_layout()
	plt.savefig(dssFileLoc + '/' + output_name)
	plt.clf


def get_hosting_cap(file_path, turb_min, turb_max, snapshot_or_timeseries):
	tree = dss_manipulation.dss_to_tree(file_path)
	loads = [y.get('object') for y in tree if y.get('object','').startswith('load.')]
	for y in tree:
		if y.get('object','').startswith('load.') and 'daily' in y.keys():
			tree = dss_manipulation.host_cap_snapshot_arrange(tree)
			break

	# adds generation at each load 15.6 kW at a time
	i  = None
	for i in range(turb_min, turb_max):
		dg_tree = dss_manipulation.add_turbine(tree, i, 15.6)
		dss_manipulation.tree_to_dss(dg_tree, 'cap_circuit.dss')
		if snapshot_or_timeseries == 'snapshot':
			dssFileLoc = os.path.dirname(os.path.abspath('cap_circuit.dss'))
			coords = runDSS('cap_circuit.dss')
			runDssCommand('Export voltages "' + dssFileLoc + '/volts.csv"')
			volts = pd.read_csv(dssFileLoc + '/volts.csv')
			volt_values = {}
			for index, row in volts.iterrows():
				volt_values[row['Bus']] = row[' pu1']
			# sort the values and break if the largest surpasses 1.05
			max_volt = max(volt_values.values())
			print(i, max_volt)
			if max_volt >= 1.05:
				network_plot("cap_circuit.dss", i)
				break
		if snapshot_or_timeseries == 'timeseries':
			maximums = newQstsPlot('cap_circuit.dss', 60, 8760)
			print(i, maximums)
			if all(i < 1.05 for i in maximums) == False:
				network_plot("cap_circuit.dss", i)
				break
	else:
		print("Circuit did not reach hosting capacity at " + str(i + 1) + " 15.6 kW turbines, or " + str(15.6 * (i + 1)) + " kW.")


def newQstsPlot(filePath, stepSizeInMinutes, numberOfSteps, keepAllFiles=False, actions={}, filePrefix='timeseries'):
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
		# elif obData.startswith('vsource.'):
		# 	runDssCommand(f'new object=monitor.{mon_name} element={obType}.{name} terminal=1 mode=0')
		# 	mon_names.append(mon_name)
		# elif obData.startswith('isource.'):
		# 	runDssCommand(f'new object=monitor.{mon_name} element={obType}.{name} terminal=1 mode=0')
		# 	mon_names.append(mon_name)
		# elif obData.startswith('generator.') or obData.startswith('isource.') or obData.startswith('storage.'):
		# 	mon_name = f'mongenerator-{name}'
		# 	runDssCommand(f'new object=monitor.{mon_name} element={obType}.{name} terminal=1 mode=1 ppolar=no')
		# 	mon_names.append(mon_name)
		elif ob.get('object','').startswith('load.'):
			runDssCommand(f'new object=monitor.{mon_name} element={obType}.{name} terminal=1 mode=0')
			mon_names.append(mon_name)
			new_kv = pd.DataFrame({'kv':[float(ob.get('kv',1.0))],'Name':['monload-' + name]})
			base_kvs = base_kvs.append(new_kv)
		# elif ob.get('object','').startswith('capacitor.'):
		# 	runDssCommand(f'new object=monitor.{mon_name} element={obType}.{name} terminal=1 mode=6')
		# 	mon_names.append(mon_name)
		# elif ob.get('object','').startswith('regcontrol.'):
		# 	tformer = ob.get('transformer','NONE')
		# 	winding = ob.get('winding',1)
		# 	runDssCommand(f'new object=monitor.{mon_name} element=transformer.{tformer} terminal={winding} mode=2')
		# 	mon_names.append(mon_name)
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
	all_gen_df = pd.DataFrame()
	all_load_df = pd.DataFrame()
	all_source_df = pd.DataFrame()
	all_control_df = pd.DataFrame()
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
			df['Name'] = name
			all_load_df = pd.concat([all_load_df, df], ignore_index=True, sort=False)
			# # pd.set_option('display.max_columns', None)
		elif name.startswith('mongenerator-'):
			df['Name'] = name
			all_gen_df = pd.concat([all_gen_df, df], ignore_index=True, sort=False)
		elif name.startswith('monvsource-'):
			df['Name'] = name
			all_source_df = pd.concat([all_source_df, df], ignore_index=True, sort=False)
		elif name.startswith('monisource-'):
			df['Name'] = name
			all_source_df = pd.concat([all_source_df, df], ignore_index=True, sort=False)
		elif name.startswith('moncapacitor-'):
			df['Type'] = 'Capacitor'
			df['Name'] = name
			df = df.rename({' Step_1 ': 'Tap(pu)'}, axis='columns') #HACK: rename to match regulator tap name
			all_control_df = pd.concat([all_control_df, df], ignore_index=True, sort=False)
		elif name.startswith('monregcontrol-'):
			df['Type'] = 'Transformer'
			df['Name'] = name
			df = df.rename({' Tap (pu)': 'Tap(pu)'}, axis='columns') #HACK: rename to match cap tap name
			all_control_df = pd.concat([all_control_df, df], ignore_index=True, sort=False)
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
	if not all_gen_df.empty:
		all_gen_df.sort_values(['Name','hour'], inplace=True)
		all_gen_df.columns = all_gen_df.columns.str.replace(r'[ "]','',regex=True)
		all_gen_df.to_csv(f'{dssFileLoc}/{filePrefix}_gen.csv', index=False)
	if not all_control_df.empty:
		all_control_df.sort_values(['Name','hour'], inplace=True)
		all_control_df.columns = all_control_df.columns.str.replace(r'[ "]','',regex=True)
		all_control_df.to_csv(f'{dssFileLoc}/{filePrefix}_control.csv', index=False)
	if not all_source_df.empty:
		all_source_df.sort_values(['Name','hour'], inplace=True)
		all_source_df.columns = all_source_df.columns.str.replace(r'[ "]','',regex=True)
		all_source_df["P1(kW)"] = all_source_df["V1"].astype(float) * all_source_df["I1"].astype(float) / 1000.0
		all_source_df["P2(kW)"] = all_source_df["V2"].astype(float) * all_source_df["I2"].astype(float) / 1000.0
		all_source_df["P3(kW)"] = all_source_df["V3"].astype(float) * all_source_df["I3"].astype(float) / 1000.0
		all_source_df.to_csv(f'{dssFileLoc}/{filePrefix}_source.csv', index=False)
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
		all_load_df.to_csv(f'{dssFileLoc}/{filePrefix}_load.csv', index=False)
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


# get_hosting_cap("lehigh.dss", 160, 190, 'timeseries')
# get_hosting_cap("wto_buses_xy.dss", 0, 10, 'snapshot')
newQstsPlot('lehigh.dss', 60, 8760)