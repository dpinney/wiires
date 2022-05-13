'''
Runs dss file and saves csv of monitor output. 
Graphs monitor output csv (3 or single phase).
'''

import os
import glob
import xarray
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import opendssdirect as dss
import fire

# dss.Basic.DataPath("./data/")


def run_dss_command(dss_cmd):
	from opendssdirect import run_command, Error
	x = run_command(dss_cmd)
	latest_error = Error.Description()
	if latest_error != '':
		print('OpenDSS Error:', latest_error)
	return x


def run_dss(dss_file):
    dss_string = open(dss_file, "r").read()
    DSSNAME = 'dss_string.dss'
    with open(DSSNAME,'w') as file:
        file.write(dss_string)
    x = run_dss_command(f'Redirect "{DSSNAME}"')
    return x


def graph_three_phase(csv_path):
	csv_data = pd.read_csv(csv_path)

	# convert voltages to  decimal portions of the nominal (2400 for 3 phase)
	unit_volts = []
	for i in range(1, 4):
		unit_volts.append([])
		for x in csv_data[f' V{i}']:
			unit_volts[i-1].append(x/2400)
		csv_data[f' V{i}'] = unit_volts[i-1]

	# overload hosting capactiy error message 
	for x in range(len(unit_volts)):
		for y in unit_volts[x]:
			if y > 1.05:
				print('Error: reached hosting capacity')

	# plot
	csv_data.plot(0, [2,4,6])
	plt.xlim([0,96])
	plt.show()


def graph_single_phase(csv_path):
	csv_data = pd.read_csv(csv_path)

	# convert voltages to decimal portions of the nominal (2400 or 280 for single phase)
	unit_volts = []
	for x in csv_data[' V1']:
	  if x > 1000:
	    unit_volts.append(x/2400)
	  else:
	    unit_volts.append(x/280)
	csv_data[' V1'] = unit_volts

	# overload hosting capactiy error message 
	for x in unit_volts:
		if x > 1.05:
			print('Error: reached hosting capacity')

	# plot
	csv_data.plot(0, [2])
	plt.xlim([0,96])
	plt.show()


if __name__ == '__main__':
	fire.Fire()


# run_dss('data/lehigh.dss')