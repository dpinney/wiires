from scipy.optimize import minimize 
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dss_manipulation import dss_to_tree
import fire
import LCEM


def OEM(load, solar_output_ds, wind_output_ds):
	solar_output_ds.reset_index(drop=True, inplace=True)
	wind_output_ds.reset_index(drop=True, inplace=True)

	# note: .csv must contain one column of 8760 values 
	if isinstance(load, str) == True:
		if load.endswith('.csv'):
			demand = pd.read_csv(load, delimiter = ',', squeeze = True)
	else:
		demand = pd.Series(load)

	merged_frame = pd.DataFrame({
	    'solar':solar_output_ds,
	    'wind':wind_output_ds,
	    'demand':demand
	    })
	merged_frame = merged_frame.fillna(0) # replaces N/A or NaN values with 0s
	merged_frame[merged_frame < 0] = 0 # no negative generation or load

	def objective(x):
		merged_frame['solar'] = x[0] * merged_frame['solar']
		merged_frame['wind'] = x[1] * merged_frame['wind']
		merged_frame['demand_minus_renewables'] = merged_frame['demand'] - (merged_frame['solar'] + merged_frame['wind'])

		rendf = pd.DataFrame(merged_frame).reset_index()
		STORAGE_DIFF = []
		for i in rendf.index:
		  prev_charge = x[2] if i == rendf.index[0] else rendf.at[i-1, 'charge'] # can set starting charge here 
		  net_renewables = rendf.at[i, 'demand_minus_renewables'] # use the existing renewable resources  
		  new_net_renewables = net_renewables - prev_charge # if positive: fossil fuel. if negative: charge battery until maximum. 
		  if new_net_renewables < 0: 
		    charge = min(-1 * new_net_renewables, x[2]) # charges battery by the amount new_net_renewables is negative until hits max 
		    rendf.at[i, 'demand_minus_renewables'] = new_net_renewables + charge # cancels out unless hits storage limit. then curtailment 
		  else:
		    charge = 0.0 # we drained the battery 
		    rendf.at[i, 'demand_minus_renewables'] = new_net_renewables # the amount of fossil we'll need 
		  rendf.at[i, 'charge'] = charge 
		  STORAGE_DIFF.append(x[2] if i == rendf.index[0] else rendf.at[i, 'charge'] - rendf.at[i-1, 'charge'])
		rendf['fossil'] = [x if x>0 else 0.0 for x in rendf['demand_minus_renewables']]
		rendf['curtailment'] = [x if x<0 else 0.0 for x in rendf['demand_minus_renewables']]

		# set energy costs in dollars per Watt 
		solar_cost = sum(rendf['solar']) * 0.000_024
		wind_cost = sum(rendf['wind']) * 0.000_009

		ABS_DIFF = [abs(i) for i in STORAGE_DIFF]
		CYCLES = sum(ABS_DIFF) * 0.5
		# multiply by LCOS 
		storage_cost = CYCLES * 0.000_055
		
		jan_demand = rendf['fossil'][0:744]
		feb_demand = rendf['fossil'][744:1416]
		mar_demand = rendf['fossil'][1416:2160]
		apr_demand = rendf['fossil'][2160:2880]
		may_demand = rendf['fossil'][2880:3624]
		jun_demand = rendf['fossil'][3624:4344]
		jul_demand = rendf['fossil'][4344:5088]
		aug_demand = rendf['fossil'][5088:5832]
		sep_demand = rendf['fossil'][5832:6552]
		oct_demand = rendf['fossil'][6552:7296]
		nov_demand = rendf['fossil'][7296:8016]
		dec_demand = rendf['fossil'][8016:8760] 
		monthly_demands = [jan_demand, feb_demand, mar_demand, apr_demand, may_demand, jun_demand, jul_demand, aug_demand, sep_demand, oct_demand, nov_demand, dec_demand]

		energy_rate = 0.000_150 # $ per Watt
		demand_rate = 20 # typical demand rate in $ per kW
		demand_rate = demand_rate / 1000
		fossil_cost = [energy_rate * sum(mon_dem) + demand_rate*max(mon_dem) for mon_dem in monthly_demands]
		fossil_cost = sum(fossil_cost)
		# fossil_cost = sum(rendf['fossil']) * 9e99
		
		tot_cost = wind_cost + solar_cost + storage_cost + fossil_cost
		return tot_cost

	x0 = [10000, 10000, 10000]
	solution = minimize(objective,x0,method='SLSQP')
	return solution.x 


def scipy_optimize_dss(file_path):
	dss_tree = dss_to_tree(file_path)
	scipy_optimize(dss_tree)


if __name__ == '__main__':
	fire.Fire()

weather_ds = LCEM.get_weather(39.952437, -75.16378, 2019)
solar_output_ds = LCEM.get_solar(weather_ds)
wind_output_ds = LCEM.get_wind(weather_ds)
results = OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds)
print(results)