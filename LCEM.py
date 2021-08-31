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
import multiprocessing
from datetime import datetime as dt, timedelta
from functools import partial


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


def clean_series(*series):
	cleaned_series = []
	for s in series:
		s = s.fillna(0) # replaces N/A or NaN values with 0s
		s[s < 0] = 0 # no negative generation or load
		# s = s.apply(lambda x : x if x > 0 else 0)
		cleaned_series.append(s)
	return cleaned_series


# def new_renewables(solar_output_ds, solar_capacity, wind_output_ds, wind_capacity):
# 	new_solar = solar_output_ds * solar_capacity 
# 	new_wind = wind_output_ds * wind_capacity
# 	new_solar, new_wind = clean_series(new_solar, new_wind)
# 	return new_solar, new_wind


def new_demand(load, new_solar, new_wind):
	# note: .csv must contain one column of 8760 values 
	if isinstance(load, str) == True:
		if load.endswith('.csv'):
			demand = pd.read_csv(load, delimiter = ',', squeeze = True)
	else:
		demand = pd.Series(load) 
	
	demand = clean_series(demand)[0]

	merged_frame = pd.DataFrame({
			'solar':new_solar,
			'wind':new_wind,
			'demand':demand
			})
	merged_frame['demand_minus_renewables'] = merged_frame['demand'] - (merged_frame['solar'] + merged_frame['wind'])
	return merged_frame['demand_minus_renewables']


# def peak_shaver(demand_after_renewables, battCapacity, battDischarge, battCharge):
# 	positive_demand = []
# 	curtailment = []
# 	for x in demand_after_renewables:
# 		if x <= 0:
# 			curtailment.append(x)
# 			positive_demand.append(0)
# 		if x > 0:
# 			curtailment.append(0)
# 			positive_demand.append(x)
# 	if battCapacity == 0:
# 		return positive_demand, curtailment, [0] * 8760, 0
# 	dates = [(dt(2019, 1, 1) + timedelta(hours=1)*x) for x in range(8760)]
# 	dc = [{'power': load, 'month': date.month -1, 'hour': date.hour} for load, date in zip(positive_demand, dates)]
# 	# list of 12 lists of monthly demands
# 	demandByMonth = [[t['power'] for t in dc if t['month']==x] for x in range(12)]
# 	monthlyPeakDemand = [max(lDemands) for lDemands in demandByMonth] 
# 	SoC = battCapacity
# 	ps = [battDischarge] * 12
# 	# keep shrinking peak shave (ps) until every month doesn't fully expend the battery
# 	while True:
# 		SoC = battCapacity 
# 		incorrect_shave = [False] * 12 
# 		for row in dc:			
# 			month = row['month']
# 			if not incorrect_shave[month]:
# 				powerUnderPeak = monthlyPeakDemand[month] - row['power'] - ps[month] 
# 				charge = (min(powerUnderPeak, battCharge, battCapacity - SoC) if powerUnderPeak > 0 
# 					else -1 * min(abs(powerUnderPeak), battDischarge, SoC))
# 				if charge == -1 * SoC: 
# 					incorrect_shave[month] = True
# 				SoC += charge 
# 				# SoC = 0 when incorrect_shave[month] == True 
# 				row['netpower'] = row['power'] + charge 
# 				row['battSoC'] = SoC
# 				if row['netpower'] > 0:
# 					row['fossil'] = row['netpower']
# 				else:
# 					row['fossil'] = 0 
# 		ps = [s-100 if incorrect else s for s, incorrect in zip(ps, incorrect_shave)]
# 		if not any(incorrect_shave):
# 			break
# 	charge = [t['battSoC'] for t in dc]
# 	capacity_times_cycles = sum([charge[i]-charge[i+1] for i, x in enumerate(charge[:-1]) if charge[i+1] < charge[i]])
# 	fossil = [t['fossil'] for t in dc]
# 	return fossil, curtailment, charge, capacity_times_cycles


# def batt_pusher(demand_after_renewables, battCapacity, battDischarge, battCharge):
# 	mix_df = pd.DataFrame(demand_after_renewables).reset_index()
# 	STORAGE_DIFF = []
# 	for i in mix_df.index:
# 			prev_charge = battCapacity if i == mix_df.index[0] else mix_df.at[i-1, 'charge'] # can set starting charge here 
# 			net_renewables = mix_df.at[i, 'demand_minus_renewables'] # use the existing renewable resources  
# 			if prev_charge > battDischarge:
# 				new_net_renewables = net_renewables - battDischarge 
# 			else:
# 				new_net_renewables = net_renewables - prev_charge # if positive: fossil fuel. if negative: charge battery until maximum. 
# 			if new_net_renewables < 0: 
# 					charge = min(-1 * new_net_renewables, battCharge) # charges battery by the amount new_net_renewables is negative until hits max chargeable in an hour
# 					mix_df.at[i, 'demand_minus_renewables'] = new_net_renewables + charge # cancels out unless hits storage limit. then curtailment # either cancels out and represents a demand perfectly met with renewables and some battery (remaining amount of battery = charge) or 
# 			else:
# 					charge = 0.0 # we drained the battery 
# 					mix_df.at[i, 'demand_minus_renewables'] = new_net_renewables # the amount of fossil we'll need (demand minus renewables minus battery discharge (max dischargeable in an hour))
# 			if battCharge < (-1 * new_net_renewables - prev_charge) and (battCapacity - prev_charge):
# 				charge = prev_charge + battCharge
# 				mix_df.at[i, 'demand_minus_renewables'] = min(-1 * new_net_renewables, battCharge) - prev_charge - battCharge
# 			mix_df.at[i, 'charge'] = charge 
# 			STORAGE_DIFF.append(0 if i == mix_df.index[0] else mix_df.at[i, 'charge'] - mix_df.at[i-1, 'charge'])
# 	mix_df['fossil'] = [x if x>0 else 0.0 for x in mix_df['demand_minus_renewables']]
# 	mix_df['curtailment'] = [x if x<0 else 0.0 for x in mix_df['demand_minus_renewables']] # TO DO: this plots incorrectly
# 	ABS_DIFF = [abs(i) for i in STORAGE_DIFF]
# 	capacity_times_cycles = (sum(ABS_DIFF) * 0.5)
# 	return mix_df['fossil'], mix_df['curtailment'], mix_df['charge'], capacity_times_cycles


# def cost_calculator(fossil_ds, curtailment_ds, solar_output_ds, wind_output_ds, capacity_times_cycles, solar_rate=.000_024, wind_rate=.000_009, batt_rate=.000_055, grid_rate=.000_070, TOU=None, demand_rate=.02, net_metering=False, export_rate=.000_040):
# 	solar_cost = sum(solar_output_ds) * solar_rate
# 	wind_cost = sum(wind_output_ds) * wind_rate
# 	storage_cost = capacity_times_cycles * batt_rate

# 	jan_demand = fossil_ds[0:744]
# 	feb_demand = fossil_ds[744:1416]
# 	mar_demand = fossil_ds[1416:2160]
# 	apr_demand = fossil_ds[2160:2880]
# 	may_demand = fossil_ds[2880:3624]
# 	jun_demand = fossil_ds[3624:4344]
# 	jul_demand = fossil_ds[4344:5088]
# 	aug_demand = fossil_ds[5088:5832]
# 	sep_demand = fossil_ds[5832:6552]
# 	oct_demand = fossil_ds[6552:7296]
# 	nov_demand = fossil_ds[7296:8016]
# 	dec_demand = fossil_ds[8016:8760]
# 	monthly_demands = [jan_demand, feb_demand, mar_demand, apr_demand, may_demand, jun_demand, jul_demand, aug_demand, sep_demand, oct_demand, nov_demand, dec_demand]

# 	if TOU != None:
# 		# note: .csv must contain one column of 8760 values 
# 		if isinstance(TOU, str) == True:
# 			if load.endswith('.csv'):
# 				TOU = pd.read_csv(TOU, delimiter = ',', squeeze = True)
# 		else:
# 			TOU = list(TOU)
# 		TOU_cost = [x * y for x, y in zip(TOU, fossil_ds)]
# 		demand_charges = [demand_rate*max(mon_dem) for mon_dem in monthly_demands]
# 		fossil_cost = sum(TOU_cost) + sum(demand_charges)
# 	else:
# 		demand_charges = [grid_rate * sum(mon_dem) + demand_rate*max(mon_dem) for mon_dem in monthly_demands]
# 		fossil_cost = sum(demand_charges)

# 	if net_metering == True:
# 		resale = sum(curtailment_ds) * export_rate
# 		tot_cost = solar_cost + wind_cost + storage_cost + fossil_cost + resale
# 	else:
# 		tot_cost = solar_cost + wind_cost + storage_cost + fossil_cost

# 	return tot_cost


def mix_graph(load, latitude, longitude, year, solar_capacity, wind_capacity, cellCapacity, peak_shave=False, 
	dischargeRate=250, chargeRate=250, cellQuantity=100, dodFactor=100, solar_rate=1600, wind_rate=2000, batt_rate=840, inverter_rate=420, grid_rate=0.11, 
	TOU=None, demand_rate=18, net_metering=True, export_rate=0.034, csv=False, output_path='test'):
	# note: .csv must contain one column of 8760 values 
	if isinstance(load, str) == True:
		if load.endswith('.csv'):
			demand = pd.read_csv(load, delimiter = ',', squeeze = True)
	else:
		demand = pd.Series(load) 
	weather_ds = get_weather(latitude, longitude, year)
	solar_output_ds = get_solar(weather_ds)
	wind_output_ds = get_wind(weather_ds)
	new_solar, new_wind = new_renewables(solar_output_ds, solar_capacity, wind_output_ds, wind_capacity)
	demand_after_renewables = new_demand(load, new_solar, new_wind)
	dodFactor = dodFactor/ 100.0
	battCapacity = cellQuantity * cellCapacity * dodFactor
	battDischarge = cellQuantity * dischargeRate 
	battCharge = cellQuantity * chargeRate
	if peak_shave == True:
		fossil, curtailment, charge, capacity_times_cycles = peak_shaver(demand_after_renewables, battCapacity, battDischarge, battCharge)
	else:
		fossil, curtailment, charge, capacity_times_cycles = batt_pusher(demand_after_renewables, battCapacity, battDischarge, battCharge)

	plotly_horiz_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
	mix_chart = go.Figure(
			[
					go.Scatter(x=demand.index, y=new_solar, name='Solar Feedin (W)', stackgroup='one', visible='legendonly'),
					go.Scatter(x=demand.index, y=new_wind, name='Wind Feedin (W)', stackgroup='one', visible='legendonly'),
					go.Scatter(x=demand.index, y=demand, name='Demand (W)', stackgroup='two', visible='legendonly'),
					go.Scatter(x=demand.index, y=fossil, name='Fossil (W)', stackgroup='one'),
					go.Scatter(x=demand.index, y=demand - fossil, name='Renewables (W)', stackgroup='one'),
					go.Scatter(x=demand.index, y=curtailment, name='Curtail Ren. (W)', stackgroup='four'),
					go.Scatter(x=demand.index, y=charge, name='Storage (SOC, Wh)', stackgroup='three', visible='legendonly'),
			],
			go.Layout(
					title = 'Combined Feed-in',
					yaxis = {'title': 'Watts'},
					legend = plotly_horiz_legend
			)
	)
	if csv == True:
		cost_calculator(fossil, curtailment, solar_capacity, wind_capacity, capacity_times_cycles, solar_rate, wind_rate, batt_rate, inverter_rate, grid_rate, TOU, demand_rate, net_metering, export_rate, True, output_path)
	return mix_chart.show()


def LCEM(load, latitude, longitude, year, solar_min, solar_max, solar_step, wind_min, wind_max, wind_step, batt_min, batt_max, batt_step, peak_shave=False, 
	dischargeRate=250, chargeRate=250, cellQuantity=100, dodFactor=100, solar_rate=.000_024, wind_rate=.000_009, batt_rate=840, inverter_rate=420, grid_rate=.000_070, TOU=None, 
	demand_rate=.02, net_metering=False, export_rate=.000_040, refined_grid_search=False, multiprocess=False, cores=8, show_mix=True, csv=True, output_path='test'):
	weather_ds = get_weather(latitude, longitude, year)
	solar_output_ds = get_solar(weather_ds)
	wind_output_ds = get_wind(weather_ds)
	results = []
	dodFactor = dodFactor/ 100.0
	battDischarge = cellQuantity * dischargeRate 
	battCharge = cellQuantity * chargeRate
	if solar_min < 0:
		solar_min = 0
	if wind_min < 0:
		wind_min = 0
	if batt_min < 0:
		batt_min = 0
	if multiprocess == True:
		solar_iter = float_range(solar_min, solar_max, solar_step)
		wind_iter = float_range(wind_min, wind_max, wind_step)
		batt_iter = float_range(batt_min, batt_max, batt_step)
		param_list = list(itertools.product(solar_iter,wind_iter,batt_iter))
		pool = multiprocessing.Pool(processes=cores)
		func = partial(multiprocessor, load, solar_output_ds, wind_output_ds, peak_shave, battDischarge, battCharge, cellQuantity, dodFactor, solar_rate, wind_rate, batt_rate, inverter_rate, grid_rate, TOU, demand_rate, net_metering, export_rate)
		print(f' Running multiprocessor {len(param_list)} times with {cores} cores')
		results.append(pool.map(func, param_list))
		results = results[0]
	else:	
		for solar in float_range(solar_min, solar_max, solar_step):
			for wind in float_range(wind_min, wind_max, wind_step):
				for batt in float_range(batt_min, batt_max, batt_step):
					new_solar, new_wind = new_renewables(solar_output_ds, solar, wind_output_ds, wind)
					demand_after_renewables = new_demand(load, new_solar, new_wind)
					battCapacity = cellQuantity * batt * dodFactor
					if peak_shave == True:
						fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = peak_shaver(demand_after_renewables, battCapacity, battDischarge, battCharge)
					else:
						fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = batt_pusher(demand_after_renewables, battCapacity, battDischarge, battCharge)
					tot_cost = cost_calculator(fossil_ds, curtailment_ds, new_solar, new_wind, capacity_times_cycles, solar_rate, wind_rate, batt_rate, inverter_rate, grid_rate, TOU, demand_rate, net_metering, export_rate)
					results.append([tot_cost, solar, wind, batt, sum(fossil_ds)])
	results.sort(key=lambda x:x[0])
	print("LCEM iteration results:", results[0:20])
	if refined_grid_search == True:
		x, y, z = solar_step, wind_step, batt_step
		while x > 1_000 and y > 1_000 and z > 1_000:
			print('Beginning recursive LCEM iteration')
			new_solar = results[0][1]
			new_wind = results[0][2]
			new_batt = results[0][3]
			print('new_solar:', new_solar, 'new_wind:', new_wind, 'new_batt:', new_batt)
			a, b, c = x * 0.9, y * 0.9, z * 0.9
			results = LCEM(load, latitude, longitude, year, new_solar - a, new_solar + a, x / 10, new_wind - b, new_wind + b, y / 10, new_batt - c, new_batt + c, z / 10, peak_shave, dischargeRate, chargeRate, cellQuantity, dodFactor, solar_rate, wind_rate, batt_rate, inverter_rate, grid_rate, TOU, demand_rate, net_metering, export_rate, False, multiprocess, cores, False)
			print(" Finshed recursive LCEM iteration")
			x, y, z = x / 10, y / 10, z / 10
	if show_mix == True:
		mix_graph(load, latitude, longitude, year, results[0][1], results[0][2], results[0][3], peak_shave, dischargeRate, chargeRate, cellQuantity, dodFactor, solar_rate, wind_rate, batt_rate, inverter_rate, grid_rate, TOU, demand_rate, net_metering, export_rate, csv, output_path)
	return results


# def multiprocessor(load, solar_output_ds, wind_output_ds, peak_shave, battDischarge, battCharge, cellQuantity, dodFactor, solar_rate, wind_rate, batt_rate, grid_rate, TOU, demand_rate, net_metering, export_rate, params):
# 	solar, wind, batt = params
# 	new_solar, new_wind = new_renewables(solar_output_ds, solar, wind_output_ds, wind)
# 	demand_after_renewables = new_demand(load, new_solar, new_wind)
# 	battCapacity = cellQuantity * batt * dodFactor
# 	if peak_shave == True:
# 		fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = peak_shaver(demand_after_renewables, battCapacity, battDischarge, battCharge)
# 	else:
# 		fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = batt_pusher(demand_after_renewables, battCapacity, battDischarge, battCharge)
# 	tot_cost = cost_calculator(fossil_ds, curtailment_ds, new_solar, new_wind, capacity_times_cycles, solar_rate, wind_rate, batt_rate, grid_rate, TOU, demand_rate, net_metering, export_rate)
# 	return tot_cost, solar, wind, batt, sum(fossil_ds)


'''
------------------------------------- modified functions for REopt comparison below -------------------------------------
'''


def new_renewables(solar_output_ds, solar_capacity, wind_output_ds, wind_capacity):
	# NOTE: DC to AC ratio is 1.2 to 1, PV inverter efficiency is 96%, and PV system losses is 14%
	new_solar = solar_output_ds * solar_capacity * (5/6) * 0.96 * 0.86
	new_wind = wind_output_ds * wind_capacity
	new_solar, new_wind = clean_series(new_solar, new_wind)
	return new_solar, new_wind


def peak_shaver(demand_after_renewables, battCapacity, battDischarge, battCharge):
	# NOTE: Battery internal efficiency is 97.5%. Inverter efficiency is 96% on both charge and discharge
	battDischarge = battDischarge * .96 * .975
	battCharge = battCharge * .96 

	positive_demand = []
	curtailment = []
	for x in demand_after_renewables:
		if x <= 0:
			curtailment.append(x)
			positive_demand.append(0)
		if x > 0:
			curtailment.append(0)
			positive_demand.append(x)
	if battCapacity == 0:
		return positive_demand, curtailment, [0] * 8760, 0
	dates = [(dt(2019, 1, 1) + timedelta(hours=1)*x) for x in range(8760)]
	dc = [{'power': load, 'month': date.month -1, 'hour': date.hour} for load, date in zip(positive_demand, dates)]
	# list of 12 lists of monthly demands
	demandByMonth = [[t['power'] for t in dc if t['month']==x] for x in range(12)]
	monthlyPeakDemand = [max(lDemands) for lDemands in demandByMonth] 
	SoC = battCapacity
	ps = [battDischarge] * 12
	# keep shrinking peak shave (ps) until every month doesn't fully expend the battery
	while True:
		SoC = battCapacity 
		incorrect_shave = [False] * 12 
		for row in dc:			
			month = row['month']
			if not incorrect_shave[month]:
				powerUnderPeak = monthlyPeakDemand[month] - row['power'] - ps[month] 
				charge = (min(powerUnderPeak, battCharge, battCapacity - SoC) if powerUnderPeak > 0 
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
		ps = [s-1000 if incorrect else s for s, incorrect in zip(ps, incorrect_shave)]
		if not any(incorrect_shave):
			break
	charge = [t['battSoC'] for t in dc]
	capacity_times_cycles = sum([charge[i]-charge[i+1] for i, x in enumerate(charge[:-1]) if charge[i+1] < charge[i]])
	fossil = [t['fossil'] for t in dc]
	return fossil, curtailment, charge, capacity_times_cycles


def batt_pusher(demand_after_renewables, battCapacity, battDischarge, battCharge):
	# NOTE: Battery internal efficiency is 97.5%. Inverter efficiency is 96% on both charge and discharge
	battDischarge = battDischarge * .96 * .975
	battCharge = battCharge * .96 

	mix_df = pd.DataFrame(demand_after_renewables).reset_index()
	STORAGE_DIFF = []
	for i in mix_df.index:
			prev_charge = battCapacity if i == mix_df.index[0] else mix_df.at[i-1, 'charge'] # can set starting charge here 
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
			if battCharge < (-1 * new_net_renewables - prev_charge) and (battCapacity - prev_charge):
				charge = prev_charge + battCharge
				mix_df.at[i, 'demand_minus_renewables'] = min(-1 * new_net_renewables, battCharge) - prev_charge - battCharge
			mix_df.at[i, 'charge'] = charge 
			STORAGE_DIFF.append(0 if i == mix_df.index[0] else mix_df.at[i, 'charge'] - mix_df.at[i-1, 'charge'])
	mix_df['fossil'] = [x if x>0 else 0.0 for x in mix_df['demand_minus_renewables']]
	mix_df['curtailment'] = [x if x<0 else 0.0 for x in mix_df['demand_minus_renewables']] # TO DO: this plots incorrectly
	ABS_DIFF = [abs(i) for i in STORAGE_DIFF]
	capacity_times_cycles = (sum(ABS_DIFF) * 0.5)
	return mix_df['fossil'], mix_df['curtailment'], mix_df['charge'], capacity_times_cycles


def cost_calculator(fossil_ds, curtailment_ds, solar_cap, wind_cap, batt_cap, solar_rate=1600, wind_rate=2000, batt_rate=840, inverter_rate=420, grid_rate=0.13, TOU=None, demand_rate=18, net_metering=False, export_rate=0.034, csv=False, output_path='test'): 
	inverter_cap = 108_300

	# Capital expenditures
	# NOTE: wind and solar federal ITC is 26% (Omitted due to REopt limitations)
	solar_cost = solar_cap * (solar_rate/1000) # * 0.74 # $/kW -> $/W
	wind_cost = wind_cap * (wind_rate /1000) # * 0.74
	storage_cost = batt_cap * (batt_rate/1000)
	# NOTE: inverter capacity below is based on REopt output on a case by case basis
	inverter_cost = inverter_cap * (inverter_rate/1000) 

	# 10 year replacement cost (batt replacement rate is 200 $/kWh, inverter replacement rate is 410 $/kW) 
	new_storage_cost = storage_cost + batt_cap * (200/1000) # $/kWh -> $/Wh
	new_inverter_cost = inverter_cost + inverter_cap * (410/1000) # $/kWh -> $/Wh

	# O&M costs (solar and wind capacities are in W and OM rates are per kW)
	solar_OM = solar_cap * (16/1000) * 25
	wind_OM = wind_cap * (40/1000) * 25

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

	demand_rate = demand_rate / 1000 # $/kW -> to $/W
	if TOU != None:
		# note: .csv must contain one column of 8760 values 
		if isinstance(TOU, str) == True:
			if load.endswith('.csv'):
				TOU = pd.read_csv(TOU, delimiter = ',', squeeze = True)
		else:
			TOU = list(TOU)
		TOU_cost = [x * y for x, y in zip(TOU, fossil_ds)]
		demand_charges = [demand_rate*max(mon_dem) for mon_dem in monthly_demands]
		fossil_cost = sum(TOU_cost) + sum(demand_charges)
	else:
		# NOTE: Annual nominal utility electricity cost escalation rate is 0.023
		grid_rate = grid_rate / 1000 # $/kWh to $/Wh
		escalation_list = [grid_rate]
		for i in range(24):
			grid_rate *= 1.023
			escalation_list.append(grid_rate)

		fossil_cost_list = []
		for rate in escalation_list:
			monthly_totals = [rate * sum(mon_dem) + demand_rate*max(mon_dem) for mon_dem in monthly_demands]
			fossil_cost_list.append(sum(monthly_totals))
		fossil_cost = sum(fossil_cost_list)

	if net_metering == True:
		export_rate = export_rate / 1000 # $/kWh -> $/Wh
		resale = sum(curtailment_ds) * export_rate * 25 # 25 years of net metering assuming no change to export rate and identical curtailment each year 
		tot_cost = solar_cost + wind_cost + new_storage_cost + new_inverter_cost + fossil_cost + solar_OM + wind_OM + resale
	else:
		tot_cost = solar_cost + wind_cost + new_storage_cost + new_inverter_cost + fossil_cost + solar_OM + wind_OM

	if csv == True: 
		cost_dict = {
		'tot_cost':tot_cost,
		'solar capacity (W)':solar_cap,
		'wind capacity (W)':wind_cap,
		'battery capacity (Wh)':batt_cap,
		'inverter capacity (W)':inverter_cap,
		'grid electricity (Wh)':sum(fossil_ds),
		'solar_cost':solar_cost,
		'wind_cost':wind_cost,
		'storage_cost':storage_cost,
		'inverter_cost':inverter_cost,
		'storage cost after replacement':new_storage_cost,
		'inverter cost after replacement':new_inverter_cost,
		'fossil_cost':fossil_cost,
		'solar_OM':solar_OM,
		'wind_OM':wind_OM, 
		'grid rates in 25 years':escalation_list,
		'12 peak demands':[max(mon_dem) for mon_dem in monthly_demands],
		'12 peak demand charges':[demand_rate*max(mon_dem) for mon_dem in monthly_demands],
		'12 grid charges':[rate * sum(mon_dem) for mon_dem in monthly_demands],
		}
		if net_metering == True:
			cost_dict['1 year of curtailment export'] = sum(curtailment_ds) * export_rate
		print(cost_dict)
		cost_df = pd.DataFrame([cost_dict])
		cost_df.to_csv(f'{output_path}.csv')
	return tot_cost


def multiprocessor(load, solar_output_ds, wind_output_ds, peak_shave, battDischarge, battCharge, cellQuantity, dodFactor, solar_rate, wind_rate, batt_rate, inverter_rate, grid_rate, TOU, demand_rate, net_metering, export_rate, params):
	solar, wind, batt = params
	new_solar, new_wind = new_renewables(solar_output_ds, solar, wind_output_ds, wind)
	demand_after_renewables = new_demand(load, new_solar, new_wind)
	battCapacity = cellQuantity * batt * dodFactor
	if peak_shave == True:
		fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = peak_shaver(demand_after_renewables, battCapacity, battDischarge, battCharge)
	else:
		fossil_ds, curtailment_ds, charge_ds, capacity_times_cycles = batt_pusher(demand_after_renewables, battCapacity, battDischarge, battCharge)
	tot_cost = cost_calculator(fossil_ds, curtailment_ds, solar, wind, batt, solar_rate, wind_rate, batt_rate, inverter_rate, grid_rate, TOU, demand_rate, net_metering, export_rate)
	return tot_cost, solar, wind, batt, sum(fossil_ds)


if __name__ == "__main__":
    LCEM('data/all_loads_vertical.csv', 39.952437, -75.16378, 2019, 0, 60_000_001, 5_000_000, 0, 60_000_001, 5_000_000, 0, 60_000_001, 5_000_000, peak_shave=True, 
    	dischargeRate=108_300, chargeRate=108_300, cellQuantity=1, dodFactor=80, solar_rate=1600, wind_rate=2000, batt_rate=840, inverter_rate=420, grid_rate=0.13, 
    	TOU=None, demand_rate=18, net_metering=True, export_rate=0.034, refined_grid_search=True, multiprocess=True, cores=8, show_mix=True, csv=True, output_path='philly_optimized_inverter')
