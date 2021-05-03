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


def runDss(dssString):
    DSSNAME = 'lehigh.dss'
    with open(DSSNAME,'w') as lehigh:
        lehigh.write(dssString)
    x = runDssCommand(f'Redirect "{DSSNAME}"')
    return x

if __name__ == '__main__':
	fire.Fire()