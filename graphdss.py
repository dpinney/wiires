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

def runDssCommand(dsscmd):
	from opendssdirect import run_command, Error
	x = run_command(dsscmd)
	latest_error = Error.Description()
	if latest_error != '':
		print('OpenDSS Error:', latest_error)
	return x


def runDss(dssFile):
    dssString = open(dssFile, "r").read()
    DSSNAME = 'dssString.dss'
    with open(DSSNAME,'w') as file:
        file.write(dssString)
    x = runDssCommand(f'Redirect "{DSSNAME}"')
    return x


def graphThreePhase(csvpath):
	zzz = pd.read_csv(csvpath)

	# convert voltages to  decimal portions of the nominal (2400 for 3 phase)
	unitVolts = []
	for i in range(1, 4):
		unitVolts.append([])
		for x in zzz[f' V{i}']:
			unitVolts[i-1].append(x/2400)
		zzz[f' V{i}'] = unitVolts[i-1]

	# plot
	zzz.plot(0, [2,4,6])
	plt.xlim([0,96])
	return plt.show()


def graphSinglePhase(csvpath):
	z = pd.read_csv(csvpath)

	# convert voltages to decimal portions of the nominal (2400 or 280 for single phase)
	unitVolts = []
	for x in z[' V1']:
	  if x > 1000:
	    unitVolts.append(x/2400)
	  else:
	    unitVolts.append(x/280)
	z[' V1'] = unitVolts

	# plot
	z.plot(0, [2])
	plt.xlim([0,96])
	return plt.show()


if __name__ == '__main__':
	fire.Fire()