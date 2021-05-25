import os
import windpowerlib
import pvlib
from windpowerlib import WindTurbine
import urllib
from feedinlib import WindPowerPlant, Photovoltaic, get_power_plant_data, era5
import glob
import xarray
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import opendssdirect as dss
import numpy as np
import warnings
import fire
import csv


solar_rate = 0.000_024 # $ per Watt
wind_rate = 0.000_009 # $ per Watt
batt_rate = 0.000_055 # $ per Watt
grid_rate = 0.000_150 # $ per Watt
demand_rate = 20 # typical demand rate in $ per kW
demand_rate = demand_rate / 1000 # $ / Watt 


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

  # https://www.labnol.org/internet/direct-links-for-google-drive/28356/ <-- how to cache 
  # cache_url = 'https://drive.google.com/uc?export=download&id=1o78EfNsd5HZMx_hZpvHiIBFDo52I1rHN' # Geographic Center of US (Lebanon, KS)
  # fname = 'ERA5_weather_data_2019_39.833333_-98.583333.nc'
  # urllib.request.urlretrieve(cache_url, fname)
  # cache_url2 = 'https://drive.google.com/uc?export=download&id=1-gPyvHNcsD7M98WC9su9mUARyMBazPhE' # City Hall, Philadelphia, PA
  # fname2 = 'ERA5_weather_data_2019_39.952437_-75.16378.nc'
  # urllib.request.urlretrieve(cache_url2, fname2)
  # cache_url3 = 'https://drive.google.com/uc?export=download&id=10TQhBScslKkm0r7SDso3hmRFBhTbp4Fp' # NRECA HQ, Arlington, VA 
  # fname3 = 'ERA5_weather_data_2019_38.8807857_-77.1130424.nc'
  # urllib.request.urlretrieve(cache_url3, fname3)
  # cache_url4 = 'https://drive.google.com/uc?export=download&id=1iCcm8MiRFYcAiTIRKdfYZqAGJ-gJJeSd' # Yuma, AZ
  # fname4 = 'ERA5_weather_data_2019_32.6056805_-114.572058.nc'
  # urllib.request.urlretrieve(cache_url4, fname4)
  # cache_url5 = 'https://drive.google.com/uc?export=download&id=1IAbr2A2oUIM5IGEFqphkt4Lc6i-PxII9' # Mt. Washington Summit, Jackson, NH
  # fname5 = 'ERA5_weather_data_2019_44.2710107_-71.3043164.nc'
  # urllib.request.urlretrieve(cache_url5, fname5)
  # cache_url6 = 'https://drive.google.com/uc?export=download&id=1Q4LnW8wue7h7ejCCtds2c1rotfbMIPX9' # Oklahoma City, OK
  # fname6 = 'ERA5_weather_data_2019_35.4676_-97.5164.nc'
  # urllib.request.urlretrieve(cache_url6, fname6)

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
  return wind_output_ds


def direct_ren_mix(load, solar, wind, batt, solar_output_ds, wind_output_ds):
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
  merged_frame['solar'] = solar * merged_frame['solar']
  merged_frame['wind'] = wind * merged_frame['wind']
  merged_frame['demand_minus_renewables'] = merged_frame['demand'] - (merged_frame['solar'] + merged_frame['wind'])

  mix_df = pd.DataFrame(merged_frame).reset_index()
  STORAGE_DIFF = []
  for i in mix_df.index:
      prev_charge = batt if i == mix_df.index[0] else mix_df.at[i-1, 'charge'] # can set starting charge here 
      net_renewables = mix_df.at[i, 'demand_minus_renewables'] # use the existing renewable resources  
      new_net_renewables = net_renewables - prev_charge # if positive: fossil fuel. if negative: charge battery until maximum. 
      if new_net_renewables < 0: 
          charge = min(-1 * new_net_renewables, batt) # charges battery by the amount new_net_renewables is negative until hits max 
          mix_df.at[i, 'demand_minus_renewables'] = new_net_renewables + charge # cancels out unless hits storage limit. then curtailment 
      else:
          charge = 0.0 # we drained the battery 
          mix_df.at[i, 'demand_minus_renewables'] = new_net_renewables # the amount of fossil we'll need 
      mix_df.at[i, 'charge'] = charge 
      STORAGE_DIFF.append(batt if i == mix_df.index[0] else mix_df.at[i, 'charge'] - mix_df.at[i-1, 'charge'])
  mix_df['fossil'] = [x if x>0 else 0.0 for x in mix_df['demand_minus_renewables']]
  mix_df['curtailment'] = [x if x<0 else 0.0 for x in mix_df['demand_minus_renewables']]

  plotly_horiz_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
  mix_chart = go.Figure(
      [
          go.Scatter(x=mix_df['index'], y=mix_df['solar'], name='Solar Feedin (W)', stackgroup='one', visible='legendonly'),
          go.Scatter(x=mix_df['index'], y=mix_df['wind'], name='Wind Feedin (W)', stackgroup='one', visible='legendonly'),
          go.Scatter(x=mix_df['index'], y=mix_df['demand'], name='Demand (W)', stackgroup='two', visible='legendonly'),
          go.Scatter(x=mix_df['index'], y=mix_df['fossil'], name='Fossil (W)', stackgroup='one'),
          go.Scatter(x=mix_df['index'], y=mix_df['demand'] - mix_df['fossil'], name='Renewables (W)', stackgroup='one'),
          go.Scatter(x=mix_df['index'], y=mix_df['curtailment'], name='Curtail Ren. (W)', stackgroup='four'),
          go.Scatter(x=mix_df['index'], y=mix_df['charge'], name='Storage (SOC, Wh)', stackgroup='three', visible='legendonly'),
      ],
      go.Layout(
          title = 'Combined Feed-in',
          yaxis = {'title': 'Watts'},
          legend = plotly_horiz_legend
      )
  )

  # set levelized energy costs in dollars per Watt hour 
  solar_cost = sum(mix_df['solar']) * solar_rate
  wind_cost = sum(mix_df['wind']) * wind_rate
  ABS_DIFF = [abs(i) for i in STORAGE_DIFF]
  CYCLES = sum(ABS_DIFF) * 0.5
  # multiply by LCOS 
  storage_cost = CYCLES * batt_rate

  jan_demand = mix_df['fossil'][0:744]
  feb_demand = mix_df['fossil'][744:1416]
  mar_demand = mix_df['fossil'][1416:2160]
  apr_demand = mix_df['fossil'][2160:2880]
  may_demand = mix_df['fossil'][2880:3624]
  jun_demand = mix_df['fossil'][3624:4344]
  jul_demand = mix_df['fossil'][4344:5088]
  aug_demand = mix_df['fossil'][5088:5832]
  sep_demand = mix_df['fossil'][5832:6552]
  oct_demand = mix_df['fossil'][6552:7296]
  nov_demand = mix_df['fossil'][7296:8016]
  dec_demand = mix_df['fossil'][8016:8760] 
  monthly_demands = [jan_demand, feb_demand, mar_demand, apr_demand, may_demand, jun_demand, jul_demand, aug_demand, sep_demand, oct_demand, nov_demand, dec_demand]

  fossil_cost = [grid_rate * sum(mon_dem) + demand_rate*max(mon_dem) for mon_dem in monthly_demands]
  fossil_cost = sum(fossil_cost)

  tot_cost = solar_cost + wind_cost + storage_cost + fossil_cost
  return mix_df[0:5], mix_chart.show(), tot_cost, sum(mix_df['fossil'])


def calc_ren_mix(load, solar, wind, batt, latitude, longitude, year):
  weather_ds = get_weather(latitude, longitude, year)
  solar_output_ds = get_solar(weather_ds)
  wind_output_ds = get_wind(weather_ds)
  mix_df, mix_chart, tot_cost, foss = direct_ren_mix(load, solar, wind, batt, solar_output_ds, wind_output_ds)
  return mix_df, mix_chart, tot_cost, foss


def direct_optimal_mix(load, solar_min, solar_max, wind_min, wind_max, batt_min, batt_max, solar_output_ds, wind_output_ds, stepsize):
  results = []
  solar_output_ds.reset_index(drop=True, inplace=True)
  wind_output_ds.reset_index(drop=True, inplace=True)

  # note: .csv must contain one column of 8760 values 
  if isinstance(load, str) == True:
  	if load.endswith('.csv'):
  		demand = pd.read_csv(load, delimiter = ',', squeeze = True)
  else:
  	demand = pd.Series(load) 

  if solar_min < 0:
  	solar_min = 0
  if wind_min < 0:
  	wind_min = 0
  if batt_min < 0:
  	batt_min = 0

  for solar in float_range(solar_min,solar_max,stepsize):
    for wind in float_range(wind_min,wind_max,stepsize):
      for batt in float_range(batt_min,batt_max,stepsize):
        merged_frame = pd.DataFrame({
            'solar':solar_output_ds,
            'wind':wind_output_ds,
            'demand':demand
            })
        merged_frame = merged_frame.fillna(0) # replaces N/A or NaN values with 0s
        merged_frame[merged_frame < 0] = 0 # no negative generation or load
        merged_frame['wind'] = wind * merged_frame['wind']
        merged_frame['solar'] = solar * merged_frame['solar']
        merged_frame['demand_minus_renewables'] = merged_frame['demand'] - (merged_frame['solar'] + merged_frame['wind'])

        rendf = pd.DataFrame(merged_frame).reset_index()
        STORAGE_DIFF = []
        for i in rendf.index:
          prev_charge = batt if i == rendf.index[0] else rendf.at[i-1, 'charge'] # can set starting charge here 
          net_renewables = rendf.at[i, 'demand_minus_renewables'] # use the existing renewable resources  
          new_net_renewables = net_renewables - prev_charge # if positive: fossil fuel. if negative: charge battery until maximum. 
          if new_net_renewables < 0: 
            charge = min(-1 * new_net_renewables, batt) # charges battery by the amount new_net_renewables is negative until hits max 
            rendf.at[i, 'demand_minus_renewables'] = new_net_renewables + charge # cancels out unless hits storage limit. then curtailment 
          else:
            charge = 0.0 # we drained the battery 
            rendf.at[i, 'demand_minus_renewables'] = new_net_renewables # the amount of fossil we'll need 
          rendf.at[i, 'charge'] = charge 
          STORAGE_DIFF.append(batt if i == rendf.index[0] else rendf.at[i, 'charge'] - rendf.at[i-1, 'charge'])
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
        
        tot_cost = wind_cost + solar_cost + storage_cost + fossil_cost
      
        results.append([tot_cost,solar,wind,batt,sum(rendf['fossil'])])
  results.sort(key=lambda x:x[0])
  return results[0:10]


def optimal_mix(load, solar_min, solar_max, wind_min, wind_max, batt_min, batt_max, latitude, longitude, year, stepsize):
	weather_ds = get_weather(latitude, longitude, year)
	solar_output_ds = get_solar(weather_ds)
	wind_output_ds = get_wind(weather_ds)
	results = direct_optimal_mix(load, solar_min, solar_max, wind_min, wind_max, batt_min, batt_max, solar_output_ds, wind_output_ds, stepsize)
	return results


def refined_LCEM(load, solar_min, solar_max, wind_min, wind_max, batt_min, batt_max, latitude, longitude, year, stepsize):
  weather_ds = get_weather(latitude, longitude, year)
  solar_output_ds = get_solar(weather_ds)
  wind_output_ds = get_wind(weather_ds)
  results = direct_optimal_mix(load, solar_min, solar_max, wind_min, wind_max, batt_min, batt_max, solar_output_ds, wind_output_ds, stepsize)
  y = stepsize
  while y > 100_000:
    new_solar = results[0][1]
    new_wind = results[0][2]
    new_batt = results[0][3]
    z = y * 0.9
    results = direct_optimal_mix(load, new_solar - z, new_solar + z, new_wind - z, new_wind + z, new_batt - z, new_batt + z, solar_output_ds, wind_output_ds, y / 10)
    y = y / 10
  return results


if __name__ == '__main__':
	fire.Fire()


capacities = refined_LCEM("./data/all_loads_vertical.csv", 0, 60_000_000, 0, 60_000_000, 0, 60_000_000, 39.952437, -75.16378, 2019, 10_000_000)
print(capacities)