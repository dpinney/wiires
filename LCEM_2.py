import os
import windpowerlib
import pvlib
from windpowerlib import WindTurbine
import urllib
from feedinlib import WindPowerPlant, Photovoltaic, get_power_plant_data, era5
import xarray
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import opendssdirect as dss
import numpy as np
import warnings
import fire
import itertools
from multiprocessing import Pool
from datetime import datetime as dt, timedelta


def safe_assert(bool_statement, error_str, keep_running):
	if keep_running:
		if not bool_statement:
			print(error_str)
	else:
		assert bool_statement, error_str


def csvValidateAndLoad(file_input, modelDir, header=0, nrows=8760, ncols=1, dtypes=[], return_type='list_by_col', ignore_nans=False, save_file=None, ignore_errors=False):
	"""
		Safely validates, loads, and saves user's file input for model's use.
		Parameters:
		file_input: stream from input_dict to be read
		modelDir: a temporary or permanent file saved to given location
		header: row of header, enter "None" if no header provided.
		nrows: skip confirmation if None
		ncols: skip confirmation if None
		dtypes: dtypes as columns should be parsed. If empty, no parsing. 
						Use "False" for column index where there should be no parsing.
						This can be used as any mapping function.
		return_type: options: 'dict', 'df', 'list_by_col', 'list_by_row'
		ignore_nans: Ignore NaN values
		save_file: if not None, save file with given *relative* pathname. It will be appended to modelDir
		ignore_errors (bool): if True, allow program to keep running when errors found and print
		Returns:
		Datatype as dictated by input parameters
	"""

	# save temporary file
	temp_path = os.path.join(modelDir, 'csv_temp.csv') if save_file == None else os.path.join(modelDir, save_file)
	with open(temp_path, 'w') as f:
		f.write(file_input)
	df = pd.read_csv(temp_path, header=header)
	
	if nrows != None:
		safe_assert( df.shape[0] == nrows, (
			f'Incorrect CSV size. Required: {nrows} rows. Given: {df.shape[0]} rows.'
		), ignore_errors)

	if ncols != None:
		safe_assert( df.shape[1] == ncols, (
			f'Incorrect CSV size. Required: {ncols} columns. Given: {df.shape[1]} columns.'
		), ignore_errors)
	
	# NaNs
	if not ignore_nans:
		d = df.isna().any().to_dict()
		nan_columns = [k for k, v in d.items() if v]
		safe_assert( 
			len(nan_columns) == 0, 
			f'NaNs detected in columns {nan_columns}. Please adjust your CSV accordingly.',
			ignore_errors
		)
	
	# parse datatypes
	safe_assert(
		(len(dtypes) == 0) or (len(dtypes) == ncols), 
		f"Length of dtypes parser must match ncols, you've entered {len(dtypes)}. If no parsing, provide empty array.",
		ignore_errors
	)
	for t, x in zip(dtypes, df.columns):
		if t != False:
			df[x] = df[x].map(lambda x: t(x))
	
	# delete file if requested
	if save_file == None:
		os.remove(temp_path)

	# return proper type
	OPTIONS = ['dict', 'df', 'list_by_col', 'list_by_row']
	safe_assert(
		return_type in OPTIONS, 
		f'return_type not recognized. Options are {OPTIONS}.',
		ignore_errors
	)
	if return_type == 'list_by_col':
		return [df[x].tolist() for x in df.columns]
	elif return_type == 'list_by_row':
		return df.values.tolist()
	elif return_type == 'df':
		return df
	elif return_type == 'dict':
		return [{k: v for k, v in row.items()} for _, row in df.iterrows()]


def float_range(start, stop, step):
	while start < stop:
		yield float(start)
		start += step


def get_weather(latitude, longitude, year):
	start_date, end_date = f'{year}-01-01', f'{year}-12-31'
	uid = '75755'
	keypart = '1997adc9-2554-4756-87e7-3b5220a44638'
	with open('./.cdsapirc','w') as keyFile:
		keyFile.write(
				'url: https://cds.climate.copernicus.eu/api/v2\n' + 
				f'key: {uid}:{keypart}'
		)
	os.system("cat './.cdsapirc'")
	os.environ['EIA_KEY'] = '431b0c60584d74a1ba22c60dbd929619'

	cache_dir = './data/'
	cache_files = os.listdir(cache_dir)
	def get_climate(latitude, longitude, year):
			cache_name = f'ERA5_weather_data_{year}_{latitude}_{longitude}.nc'
			if cache_name not in cache_files:
					print(f'Getting new ERA5 data for {cache_name}')
					weather_ds = era5.get_era5_data_from_datespan_and_position(
							variable='feedinlib',
							start_date=start_date,
							end_date=end_date,
							latitude=latitude,
							longitude=longitude,
							target_file=cache_dir + cache_name
					)
			weather_ds = xarray.open_dataset(cache_dir + cache_name)
			# weather_df = weather_ds.to_dataframe()
			# weather_df = pd.DataFrame(weather_df.to_records()) # flatten hierarchical index
			return weather_ds
	weather_ds = get_climate(latitude, longitude, year)
	return weather_ds


def get_solar(weather_ds):
	module_df = get_power_plant_data(dataset='sandiamod')
	inverter_df = get_power_plant_data(dataset='cecinverter')
	system_data = {
			'module_name': 'Advent_Solar_Ventura_210___2008_',  # module name as in database
			'inverter_name': 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',  # inverter name as in database
			'azimuth': 180,
			'tilt': weather_ds.coords.to_index()[0][0],
			'albedo': 0.2,
	}
	pv_system = Photovoltaic(**system_data)
	pvlib_df = era5.format_pvlib(weather_ds)
	pvlib_df = pvlib_df.droplevel([1,2]) # HACK: feedinlib confused by their own multiindex
	with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			solar_feedin_ac = pv_system.feedin(
					weather=pvlib_df,
					location=(weather_ds.coords.to_index()[0][0], weather_ds.coords.to_index()[0][1]),
					scaling='area')
	solar_output_ds = solar_feedin_ac / solar_feedin_ac.max()
	solar_output_ds.reset_index(drop=True, inplace=True)
	return solar_output_ds


def get_wind(weather_ds):
	bergey_turbine_data = {
			'nominal_power': 15600,  # in W
			'hub_height': 24,  # in m  
			'power_curve': pd.DataFrame(
							# https://github.com/wind-python/windpowerlib <-- for info on adding custom loadshapes 
							data={'value': [p * 1000 for p in [
												0, 0, 0.108, 0.679, 2.074, 3.824, 6.089, 8.500, 11.265, 13.664, 15.612, 16.876, 18.212, 19.096, 20.355, 20.611, 19.687]],  # kW -> W
										'wind_speed': [1.0, 2.01, 2.99, 4.01, 5.00, 6.00, 7.00, 8.00, 9.00, 9.99, 11.01, 11.97, 12.99, 13.99, 15.00, 15.97, 16.47]})  # in m/s
			}
	bergey_turbine = WindTurbine(**bergey_turbine_data)

	wind_turbine = WindPowerPlant(**bergey_turbine_data)
	windpowerlib_df = era5.format_windpowerlib(weather_ds)  
	windpowerlib_df = windpowerlib_df.droplevel([1,2])
	wind_output_ds = wind_turbine.feedin(
			weather=windpowerlib_df,
			density_correction=True,
			scaling='nominal_power',
	)
	wind_output_ds.reset_index(drop=True, inplace=True)
	return wind_output_ds


def demand_after_renewables(load, solar_output_ds, wind_output_ds, solar_capacity, wind_capacity):
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
	merged_frame['solar'] = solar_capacity * merged_frame['solar']
	merged_frame['wind'] = wind_capacity * merged_frame['wind']
	merged_frame['demand_minus_renewables'] = merged_frame['demand'] - (merged_frame['solar'] + merged_frame['wind'])
	return merged_frame['demand_minus_renewables']


def peak_shaver(demand_after_renewables, dischargeRate=250, chargeRate=250, cellQuantity=100, cellCapacity=350, dodFactor=100):
	positive_demand = []
	curtailment = []
	for x in demand_after_renewables:
		if x <= 0:
			curtailment.append(x)
			positive_demand.append(0)
		if x > 0:
			curtailment.append(0)
			positive_demand.append(x)
	dodFactor = dodFactor/ 100.0
	batt_capacity = cellQuantity * cellCapacity * dodFactor
	battDischarge = cellQuantity * dischargeRate 
	battCharge = cellQuantity * chargeRate 
	dates = [(dt(2019, 1, 1) + timedelta(hours=1)*x) for x in range(8760)]
	dc = [{'power': load, 'month': date.month -1, 'hour': date.hour} for load, date in zip(positive_demand, dates)]
	ps = [battDischarge] * 12
	# list of 12 lists of monthly demands
	demandByMonth = [[t['power'] for t in dc if t['month']==x] for x in range(12)]
	monthlyPeakDemand = [max(lDemands) for lDemands in demandByMonth]
	# keep shrinking peak shave (ps) until every month doesn't fully expend the battery
	while True:
		SoC = batt_capacity 
		incorrect_shave = [False] * 12 
		for row in dc:			
			month = row['month']
			if not incorrect_shave[month]:
				powerUnderPeak = monthlyPeakDemand[month] - row['power'] - ps[month] 
				charge = (min(powerUnderPeak, battCharge, batt_capacity - SoC) if powerUnderPeak > 0 
					else -1 * min(abs(powerUnderPeak), battDischarge, SoC))
				if charge == -1 * SoC: 
					incorrect_shave[month] = True
				SoC += charge 
				# SoC = 0 when incorrect_shave[month] == True 
				row['netpower'] = row['power'] + charge 
				row['battSoC'] = SoC
				if row['netpower'] > 0:
					row['fossil'] = row['netpower']
				else:
					row['fossil'] = 0 
		ps = [s-1_000 if incorrect else s for s, incorrect in zip(ps, incorrect_shave)]
		if not any(incorrect_shave):
			break
	capacity_times_cycles = sum([SoC[i]-SoC[i+1] for i, x in enumerate(SoC[:-1]) if SoC[i+1] < SoC[i]])
	return dc['fossil'], pd.Series(curtailment), dc['battSoC'], capacity_times_cycles


def batt_pusher(demand_after_renewables, dischargeRate=250, chargeRate=250, cellQuantity=100, cellCapacity=350, dodFactor=100):
	dodFactor = dodFactor/ 100.0
	batt_capacity = cellQuantity * cellCapacity * dodFactor
	battDischarge = cellQuantity * dischargeRate 
	battCharge = cellQuantity * chargeRate 

	mix_df = pd.DataFrame(demand_after_renewables).reset_index()
	STORAGE_DIFF = []
	for i in mix_df.index:
			prev_charge = batt_capacity if i == mix_df.index[0] else mix_df.at[i-1, 'charge'] # can set starting charge here 
			net_renewables = mix_df.at[i, 'demand_minus_renewables'] # use the existing renewable resources  
			if prev_charge > battDischarge:
				new_net_renewables = net_renewables - battDischarge 
			else:
				new_net_renewables = net_renewables - prev_charge # if positive: fossil fuel. if negative: charge battery until maximum. 
			if new_net_renewables < 0: 
					charge = min(-1 * new_net_renewables, battCharge) # charges battery by the amount new_net_renewables is negative until hits max chargeable in an hour
					mix_df.at[i, 'demand_minus_renewables'] = new_net_renewables + charge # cancels out unless hits storage limit. then curtailment # either cancels out and represents a demand perfectly met with renewables and some battery (remaining amount of battery = charge) or 
			else:
					charge = 0.0 # we drained the battery 
					mix_df.at[i, 'demand_minus_renewables'] = new_net_renewables # the amount of fossil we'll need (demand minus renewables minus battery discharge (max dischargeable in an hour))
			if battCharge < (-1 * new_net_renewables - prev_charge) and (batt_capacity - prev_charge):
				charge = prev_charge + battCharge
				mix_df.at[i, 'demand_minus_renewables'] = min(-1 * new_net_renewables, battCharge) - prev_charge - battCharge
			mix_df.at[i, 'charge'] = charge 
			STORAGE_DIFF.append(0 if i == mix_df.index[0] else mix_df.at[i, 'charge'] - mix_df.at[i-1, 'charge'])
	mix_df['fossil'] = [x if x>0 else 0.0 for x in mix_df['demand_minus_renewables']]
	mix_df['curtailment'] = [x if x<0 else 0.0 for x in mix_df['demand_minus_renewables']] # TO DO: this plots incorrectly
	ABS_DIFF = [abs(i) for i in STORAGE_DIFF]
	capacity_times_cycles = (sum(ABS_DIFF) * 0.5)
	return mix_df['fossil'], mix_df['curtailment'], mix_df['charge'], capacity_times_cycles


def cost_calculator(fossil_ds, curtailment_ds, solar_output_ds, wind_output_ds, capacity_times_cycles, solar_rate=.000_024, wind_rate=.000_009, batt_rate=.000_055, grid_rate=.000_070, demand_rate=.02, net_metering=False, export_rate=.000_040):
	solar_cost = sum(solar_output_ds) * solar_rate
	wind_cost = sum(wind_output_ds) * wind_rate
	storage_cost = capacity_times_cycles * batt_rate

	jan_demand = fossil_ds[0:744]
	feb_demand = fossil_ds[744:1416]
	mar_demand = fossil_ds[1416:2160]
	apr_demand = fossil_ds[2160:2880]
	may_demand = fossil_ds[2880:3624]
	jun_demand = fossil_ds[3624:4344]
	jul_demand = fossil_ds[4344:5088]
	aug_demand = fossil_ds[5088:5832]
	sep_demand = fossil_ds[5832:6552]
	oct_demand = fossil_ds[6552:7296]
	nov_demand = fossil_ds[7296:8016]
	dec_demand = fossil_ds[8016:8760]
	monthly_demands = [jan_demand, feb_demand, mar_demand, apr_demand, may_demand, jun_demand, jul_demand, aug_demand, sep_demand, oct_demand, nov_demand, dec_demand]
	fossil_cost = [grid_rate * sum(mon_dem) + demand_rate*max(mon_dem) for mon_dem in monthly_demands]
	fossil_cost = sum(fossil_cost)

	if net_metering == True:
		resale = sum(curtailment_ds) * export_rate
		tot_cost = solar_cost + wind_cost + storage_cost + fossil_cost + resale
	if net_metering == False:
		tot_cost = solar_cost + wind_cost + storage_cost + fossil_cost

	return tot_cost


def mix_graph(load, latitude, longitude, year, solar_capacity, wind_capacity, batt_capacity, peak_shave=False, dischargeRate=250, chargeRate=250, cellQuantity=100, dodFactor=100):
	# note: .csv must contain one column of 8760 values 
	if isinstance(load, str) == True:
		if load.endswith('.csv'):
			demand = pd.read_csv(load, delimiter = ',', squeeze = True)
	else:
		demand = pd.Series(load) 

	weather_ds = get_weather(latitude, longitude, year)
	solar_output_ds = get_solar(weather_ds)
	wind_output_ds = get_wind(weather_ds)
	demand_after_renewables_ds = demand_after_renewables(load, solar_output_ds, wind_output_ds, solar_capacity, wind_capacity)
	solar_output_ds = [x * solar_capacity for x in solar_output_ds]
	wind_output_ds = [x * wind_capacity for x in wind_output_ds]
	if peak_shave == True:
		fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = peak_shaver(demand_after_renewables_ds, dischargeRate, chargeRate, cellQuantity, batt_capacity, dodFactor)
		print('Shaved a peak!')
	else:
		fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = batt_pusher(demand_after_renewables_ds, dischargeRate, chargeRate, cellQuantity, batt_capacity, dodFactor)
		print('Pushed some batt!')

	plotly_horiz_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
	mix_chart = go.Figure(
			[
					go.Scatter(x=demand.index, y=solar_output_ds, name='Solar Feedin (W)', stackgroup='one', visible='legendonly'),
					go.Scatter(x=demand.index, y=wind_output_ds, name='Wind Feedin (W)', stackgroup='one', visible='legendonly'),
					go.Scatter(x=demand.index, y=demand, name='Demand (W)', stackgroup='two', visible='legendonly'),
					go.Scatter(x=demand.index, y=fossil_ds, name='Fossil (W)', stackgroup='one'),
					go.Scatter(x=demand.index, y=demand - fossil_ds, name='Renewables (W)', stackgroup='one'),
					go.Scatter(x=demand.index, y=curtailment_ds, name='Curtail Ren. (W)', stackgroup='four'),
					go.Scatter(x=demand.index, y=charge_ds, name='Storage (SOC, Wh)', stackgroup='three', visible='legendonly'),
			],
			go.Layout(
					title = 'Combined Feed-in',
					yaxis = {'title': 'Watts'},
					legend = plotly_horiz_legend
			)
	)
	return mix_chart.show()


def LCEM(load, latitude, longitude, year, solar_min, solar_max, solar_step, wind_min, wind_max, wind_step, batt_min, batt_max, batt_step, peak_shave = False, dischargeRate=250, chargeRate=250, cellQuantity=100, dodFactor=100, solar_rate=.000_024, wind_rate=.000_009, batt_rate=.000_055, grid_rate=.000_070, demand_rate=.02, net_metering=False, export_rate=.000_040, refined_grid_search=False):
	weather_ds = get_weather(latitude, longitude, year)
	solar_output_ds = get_solar(weather_ds)
	wind_output_ds = get_wind(weather_ds)
	results = []
	for solar in float_range(solar_min, solar_max, solar_step):
		for wind in float_range(wind_min, wind_max, wind_step):
			for batt in float_range(batt_min, batt_max, batt_step):
				demand_after_renewables_ds = demand_after_renewables(load, solar_output_ds, wind_output_ds, solar, wind)
				if peak_shave == True:
					fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = peak_shaver(demand_after_renewables_ds, dischargeRate, chargeRate, cellQuantity, batt, dodFactor)
					print('Shaved a peak!')
				else:
					fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = batt_pusher(demand_after_renewables_ds, dischargeRate, chargeRate, cellQuantity, batt, dodFactor)
					print('Pushed some batt!')
				tot_cost = cost_calculator(fossil_ds, curtailment_ds, solar_output_ds, wind_output_ds, capacity_times_cycles, solar_rate, wind_rate, batt_rate, grid_rate, demand_rate, net_metering, export_rate)
				results.append([tot_cost, solar, wind, batt, sum(fossil_ds)])
	results.sort(key=lambda x:x[0])
	print("Finished LCEM iteration")
	if refined_grid_search == True:
		x, y, z = solar_step, wind_step, batt_step
		while x > 1_000_000 and y > 1_000_000 and z > 100_000:
			new_solar = results[0][1]
			new_wind = results[0][2]
			new_batt = results[0][3]
			a, b, c = x * 0.9, y * 0.9, z * 0.9
			results = LCEM(load, latitude, longitude, year, new_solar - a, new_solar + a, x / 10, new_wind - b, new_wind + b, y / 10, new_batt - c, new_batt + c, z / 10, peak_shave, dischargeRate, chargeRate, cellQuantity, dodFactor, solar_rate, wind_rate, batt_rate, grid_rate, demand_rate, net_metering, export_rate, refined_grid_search)
			print("Finshed recursive LCEM iteration")
			x, y, z = x / 10, y / 10, z / 10
	return results[0:20]




test = LCEM('data/all_loads_vertical.csv', 39.952437, -75.16378, 2019, 0, 60_000_000, 30_000_000, 0, 60_000_000, 30_000_000, 0, 10_000_000, 5_000_000, peak_shave=False, dischargeRate=60_000_000, chargeRate=60_000_000, cellQuantity=1, dodFactor=100, solar_rate=.000_024, wind_rate=.000_009, batt_rate=.000_055, grid_rate=.000_070, demand_rate=.02, net_metering=True, export_rate=.000_040, refined_grid_search=False)
print(test)
mix_graph('data/all_loads_vertical.csv', 39.952437, -75.16378, 2019, 30_000_000, 30_000_000, 60_000_000, peak_shave=False)

