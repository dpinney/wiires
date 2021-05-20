# wiires
Wind Integration Into Rural Energy Systems

# Installation

`pip install git+https://github.com/dpinney/wiires`

# Usage Examples
```
>>> import wiires

>>> # convert a dss file to an object for manipulation 
>>> tree = wiires.dss_manipulation.dss_to_tree('lehigh.dss')
>>> tree
[OrderedDict([('!CMD', 'clear')]), OrderedDict([('!CMD', 'set'), ('defaultbasefrequency', '60')]), ...

>>> # add 2 15.6 kW wind turbines to each load
>>> tree_turb = wiires.dss_manipulation.add_turbine(tree, 2, '15.6') 

>>> # add a monitor at each load and substation 
>>> tree_turb_mon = wiires.dss_manipulation.add_monitor(tree_turb)

>>> # remove duplicate objects from tree (export statements, monitors, etc.)
>>> tree_turb_mon_cleaned = wiires.dss_manipulation.remove_dups(tree_turb_mon)

>>> # convert tree object back to dss and save to local directory 
>>> wiires.dss_manipulation.tree_to_dss(tree_turb_mon_cleaned, "./my_circuit.dss")

>>> # run dss file and save one csv per monitor locally (csv charts voltages, currents, and angles)
>>> wiires.graph_dss.run_dss("./my_circuit.dss") 
clear
set defaultbasefrequency=60 ...

>>> # graph a three phase voltage chart 
>>> wiires.graph_dss.graph_three_phase("./monitor1_output.csv")
```
![three_phase](https://user-images.githubusercontent.com/65563537/117372408-7455b200-ae97-11eb-94ed-e808a17a4710.png)
```
>>> # graph a single phase voltage chart 
>>> wiires.graph_dss.graph_single_phase("./monitor2_output.csv")
```
![single_phase](https://user-images.githubusercontent.com/65563537/117373222-d367f680-ae98-11eb-911e-ce58068be35b.png)
```
>>> # get a year of weather data for any set of coordinates starting at 1950. Hourly data starts at 1970
>>> weather_ds = LCEM.get_weather(39.952437, -75.16378, 2019)
>>> weather_ds
<xarray.Dataset>
Dimensions:    (latitude: 1, longitude: 1, time: 8760)
Coordinates:
  * longitude  (longitude) float32 -75.25
  * latitude   (latitude) float32 40.0
  * time       (time) datetime64[ns] 2019-01-01 ... 2019-12-31T23:00:00
Data variables:
    u100       (time, latitude, longitude) float32 ...
    v100       (time, latitude, longitude) float32 ...
    fsr        (time, latitude, longitude) float32 ...
    sp         (time, latitude, longitude) float32 ...
    fdir       (time, latitude, longitude) float32 ...
    ssrd       (time, latitude, longitude) float32 ...
    t2m        (time, latitude, longitude) float32 ...
    u10        (time, latitude, longitude) float32 ...
    v10        (time, latitude, longitude) float32 ...
Attributes:
    Conventions:  CF-1.6
    history:      2021-02-05 19:58:43 GMT by grib_to_netcdf-2.16.0: /opt/ecmw...

>>> # get one year of solar output in hourly Watts 
>>> solar_output_ds = LCEM.get_solar(weather_ds)
>>> solar_output_ds
time
2018-12-31 23:30:00+00:00   -0.000342
2019-01-01 00:30:00+00:00   -0.000342
2019-01-01 01:30:00+00:00   -0.000342
2019-01-01 02:30:00+00:00   -0.000342
2019-01-01 03:30:00+00:00   -0.000342
                               ...   
2019-12-31 18:30:00+00:00    0.608260
2019-12-31 19:30:00+00:00    0.356264
2019-12-31 20:30:00+00:00    0.238211
2019-12-31 21:30:00+00:00         NaN
2019-12-31 22:30:00+00:00   -0.000342
Length: 8760, dtype: float64

>>> # get one year of wind output in hourly Watts 
>>> wind_output_ds = LCEM.get_wind(weather_ds)
>>> wind_output_ds
time
2018-12-31 23:00:00+00:00    0.196656
2019-01-01 00:00:00+00:00    0.109398
2019-01-01 01:00:00+00:00    0.036799
2019-01-01 02:00:00+00:00    0.033218
2019-01-01 03:00:00+00:00    0.052314
                               ...   
2019-12-31 18:00:00+00:00    0.365715
2019-12-31 19:00:00+00:00    0.255655
2019-12-31 20:00:00+00:00    0.167706
2019-12-31 21:00:00+00:00    0.079960
2019-12-31 22:00:00+00:00    0.154031
Name: feedin_power_plant, Length: 8760, dtype: float64

>>> # find the lowest cost energy mix using grid search for a year of hourly load shapes given a location and year
>>> # function accepts loads (csv or list), max solar, wind, and storage capacity in Watts, latitude, longitude, year, and step size for grid search
>>> # solar, wind, and grid electricity costs as well as demand charge are preset but can be changed in code
>>> LCEM = wiires.LCEM.optimal_mix("./loads.csv", 60_000_000, 60_000_000, 60_000_000, 39.952437, -75.16378, 2019, 5_000_000)
>>> LCEM
[792.3018640684181, 10000, 10000, 55000, 26211.31538862757]

>>> # get the data frame of storage/curtailed generation/renewables/fossil/demand/wind/solar levels hourly for a particular preset of solar/wind/storage capacity for a particular location and year, graph the mix, get the total cost of the system, and get the total wattage of grid electricity used in the year 
>>> df, chart, total_cost, fossil_total = wiires.LCEM.calc_ren_mix("./loads.csv", 10_000_000, 10_000_000, 55_000_000, 39.952437, -75.16378, 2019)
```
![LCEM_graph](https://user-images.githubusercontent.com/65563537/117373430-32c60680-ae99-11eb-83e0-146f3efb739e.png)
```
>>> df
   index  solar         wind  ...        charge  fossil  curtailment
0      0    0.0  1966.564244  ...  55000.000000     0.0 -1175.116231
1      1    0.0  1093.980222  ...  55000.000000     0.0  -303.968290
2      2    0.0   367.990938  ...  54578.080055     0.0     0.000000
3      3    0.0   332.184239  ...  54079.690885     0.0     0.000000
4      4    0.0   523.140147  ...  53747.162622     0.0     0.000000
>>> total_cost
792.3018640684181
>>> fossil_total
26211.31538862757

>>> # a refined grid search gets the lowest cost energy mix down to the nearest Watt of capacity 
>>> ref_mix = refined_LCEM('./loads.csv', 0, 60_000, 0, 60_000, 0, 60_000, 39.952437, -75.16378, 2019, 10_000)
>>> ref_mix[0]
[787.9182973867388, 410.0, 3480.0, 0.0, 9547724.910232382]

>>> # use SciPy optimization to find the lowest cost energy mix
>>> # outputs total cost in dollars, solar, wind, and storage capacity in Watts, and grid electricity use in Wh
>>> weather_ds = LCEM.get_weather(39.952437, -75.16378, 2019)
>>> solar_output_ds = LCEM.get_solar(weather_ds)
>>> wind_output_ds = LCEM.get_wind(weather_ds)
>>> results = OEM("./loads.csv", solar_output_ds, wind_output_ds)
>>> results
[1617936.434096718, 3853494.8130528317, 6297853.202623451, 2888808.172369332, 5566520009.556752]
```