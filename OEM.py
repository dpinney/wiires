from scipy.optimize import minimize 
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dss_manipulation import dss_to_tree
import fire
import LCEM


solar_rate = 0.000_024 # $ per Watt
wind_rate = 0.000_009 # $ per Watt
batt_rate = 0.000_055 # $ per Watt
grid_rate = 0.000_070 # $ per Watt
# grid_rate = 9e99 
demand_rate = 20 # typical demand rate in $ per kW
demand_rate = demand_rate / 1000 # $ / Watt	


# def get_OEM(load, solar_max, wind_max, batt_max, solar_output_ds, wind_output_ds, stepsize):
def get_OEM(load, solar_output_ds, wind_output_ds):
	solar_output_ds.reset_index(drop=True, inplace=True)
	wind_output_ds.reset_index(drop=True, inplace=True)

	# note: .csv must contain one column of 8760 values 
	if isinstance(load, str) == True:
		if load.endswith('.csv'):
			demand = pd.read_csv(load, delimiter = ',', squeeze = True)
	else:
		demand = pd.Series(load)

	def objective(x):
		merged_frame = pd.DataFrame({
		    'solar':solar_output_ds,
		    'wind':wind_output_ds,
		    'demand':demand
		    })
		merged_frame = merged_frame.fillna(0) # replaces N/A or NaN values with 0s
		merged_frame[merged_frame < 0] = 0 # no negative generation or load
		merged_frame['solar'] = x[0] * merged_frame['solar']
		merged_frame['wind'] = x[1] * merged_frame['wind']
		merged_frame['demand_minus_renewables'] = merged_frame['demand'] - (merged_frame['solar'] + merged_frame['wind'])

		rendf = pd.DataFrame(merged_frame).reset_index()
		STORAGE_DIFF = []
		for i in rendf.index:
		  prev_charge = 0 if i == rendf.index[0] else rendf.at[i-1, 'charge'] # can set starting charge here 
		  net_renewables = rendf.at[i, 'demand_minus_renewables'] # use the existing renewable resources  
		  new_net_renewables = net_renewables - prev_charge # if positive: fossil fuel. if negative: charge battery until maximum. 
		  if new_net_renewables < 0: 
		    charge = min(-1 * new_net_renewables, x[2]) # charges battery by the amount new_net_renewables is negative until hits max 
		    rendf.at[i, 'demand_minus_renewables'] = new_net_renewables + charge # cancels out unless hits storage limit. then curtailment 
		  else:
		    charge = 0.0 # we drained the battery 
		    rendf.at[i, 'demand_minus_renewables'] = new_net_renewables # the amount of fossil we'll need 
		  rendf.at[i, 'charge'] = charge 
		  STORAGE_DIFF.append(0 if i == rendf.index[0] else rendf.at[i, 'charge'] - rendf.at[i-1, 'charge'])
		rendf['fossil'] = [x if x>0 else 0.0 for x in rendf['demand_minus_renewables']]
		rendf['curtailment'] = [x if x<0 else 0.0 for x in rendf['demand_minus_renewables']]

		# set energy costs in dollars per Watt 
		solar_cost = sum(rendf['solar']) * solar_rate
		wind_cost = sum(rendf['wind']) * wind_rate

		ABS_DIFF = [abs(i) for i in STORAGE_DIFF]
		CYCLES = sum(ABS_DIFF) * 0.5
		# multiply by LCOS 
		storage_cost = CYCLES * batt_rate
		
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

		fossil_cost = [grid_rate * sum(mon_dem) + demand_rate*max(mon_dem) for mon_dem in monthly_demands]
		fossil_cost = sum(fossil_cost)
		# fossil_cost = sum(rendf['fossil']) * 9e99
		
		return wind_cost + solar_cost + storage_cost + fossil_cost 

	# call optimization function 
	# initial_guesses = LCEM.direct_optimal_mix(load, 0, solar_max, 0, wind_max, 0, batt_max, solar_output_ds, wind_output_ds, stepsize)
	# x0 = [initial_guesses[0][1], initial_guesses[0][2], initial_guesses[0][3]]
	x0 = [1, 1, 1]
	bounds = [(0, np.inf), (0, np.inf), (0, np.inf)]
	solution = minimize(objective,x0,method='Powell',bounds=bounds)
	print(solution)

	# redefine variables for plotting 
	merged_frame = pd.DataFrame({
	    'solar':solar_output_ds,
	    'wind':wind_output_ds,
	    'demand':demand
	    })
	merged_frame = merged_frame.fillna(0) # replaces N/A or NaN values with 0s
	merged_frame[merged_frame < 0] = 0 # no negative generation or load
	merged_frame['solar'] = solution.x[0] * merged_frame['solar']
	merged_frame['wind'] = solution.x[1] * merged_frame['wind']
	merged_frame['demand_minus_renewables'] = merged_frame['demand'] - (merged_frame['solar'] + merged_frame['wind'])

	rendf = pd.DataFrame(merged_frame).reset_index()
	STORAGE_DIFF = []
	for i in rendf.index:
	  prev_charge = 0 if i == rendf.index[0] else rendf.at[i-1, 'charge'] # can set starting charge here 
	  net_renewables = rendf.at[i, 'demand_minus_renewables'] # use the existing renewable resources  
	  new_net_renewables = net_renewables - prev_charge # if positive: fossil fuel. if negative: charge battery until maximum. 
	  if new_net_renewables < 0: 
	    charge = min(-1 * new_net_renewables, solution.x[2]) # charges battery by the amount new_net_renewables is negative until hits max 
	    rendf.at[i, 'demand_minus_renewables'] = new_net_renewables + charge # cancels out unless hits storage limit. then curtailment 
	  else:
	    charge = 0.0 # we drained the battery 
	    rendf.at[i, 'demand_minus_renewables'] = new_net_renewables # the amount of fossil we'll need 
	  rendf.at[i, 'charge'] = charge 
	  STORAGE_DIFF.append(0 if i == rendf.index[0] else rendf.at[i, 'charge'] - rendf.at[i-1, 'charge'])
	rendf['fossil'] = [x if x>0 else 0.0 for x in rendf['demand_minus_renewables']]
	rendf['curtailment'] = [x if x<0 else 0.0 for x in rendf['demand_minus_renewables']]

	plotly_horiz_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
	mix_chart = go.Figure(
	    [
	        go.Scatter(x=rendf['index'], y=rendf['solar'], name='Solar Feedin (W)', stackgroup='one', visible='legendonly'),
	        go.Scatter(x=rendf['index'], y=rendf['wind'], name='Wind Feedin (W)', stackgroup='one', visible='legendonly'),
	        go.Scatter(x=rendf['index'], y=rendf['demand'], name='Demand (W)', stackgroup='two', visible='legendonly'),
	        go.Scatter(x=rendf['index'], y=rendf['fossil'], name='Fossil (W)', stackgroup='one'),
	        go.Scatter(x=rendf['index'], y=rendf['demand'] - rendf['fossil'], name='Renewables (W)', stackgroup='one'),
	        go.Scatter(x=rendf['index'], y=rendf['curtailment'], name='Curtail Ren. (W)', stackgroup='four'),
	        go.Scatter(x=rendf['index'], y=rendf['charge'], name='Storage (SOC, Wh)', stackgroup='three', visible='legendonly'),
	    ],
	    go.Layout(
	        title = 'Combined Feed-in',
	        yaxis = {'title': 'Watts'},
	        legend = plotly_horiz_legend
	    )
	).show()

	# set energy costs in dollars per Watt 
	solar_cost = sum(rendf['solar']) * solar_rate
	wind_cost = sum(rendf['wind']) * wind_rate

	ABS_DIFF = [abs(i) for i in STORAGE_DIFF]
	CYCLES = sum(ABS_DIFF) * 0.5
	# multiply by LCOS 
	storage_cost = CYCLES * batt_rate
	
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

	# $ per Watt	
	fossil_cost = [grid_rate * sum(mon_dem) + demand_rate*max(mon_dem) for mon_dem in monthly_demands]
	fossil_cost = sum(fossil_cost)
	# fossil_cost = sum(rendf['fossil']) * 9e99
	
	tot_cost = wind_cost + solar_cost + storage_cost + fossil_cost
	return [tot_cost, solution.x[0], solution.x[1], solution.x[2], sum(rendf['fossil'])] 


if __name__ == '__main__':
	fire.Fire()

weather_ds = LCEM.get_weather(39.952437, -75.16378, 2019)
solar_output_ds = LCEM.get_solar(weather_ds)
wind_output_ds = LCEM.get_wind(weather_ds)
# results = get_OEM("./data/all_loads_vertical.csv", 60_000_000, 60_000_000, 60_000_000, solar_output_ds, wind_output_ds, 5_000_000)
# item_1 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_024, 0.000_009, 0.000_055, 0.000_020, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# item_2 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_024, 0.000_009, 0.000_055, 0.000_060, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# item_3 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_024, 0.000_009, 0.000_055, 0.000_070, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# item_4 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_024, 0.000_009, 0.000_055, 0.000_150, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# item_5 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_032, 0.000_043, 0.000_087, 0.000_020, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# item_6 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_032, 0.000_043, 0.000_087, 0.000_060, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# item_7 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_032, 0.000_043, 0.000_087, 0.000_070, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# item_8 = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds, 0.000_032, 0.000_043, 0.000_087, 0.000_150, 60_000_000, 60_000_000, 60_000_000, 5_000_000)
# print(item_1, item_2, item_3, item_4, item_5, item_6, item_7, item_8)
results = get_OEM("./data/all_loads_vertical.csv", solar_output_ds, wind_output_ds)
print(results)