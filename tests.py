#wiires.LCEM.get_weather(...)

# see https://github.com/dpinney/eznlp#usage-examples for an example of how a set of tests can show users how to use the library.
import wiires

# convert a dss file to an object for manipulation 
tree = wiires.dss_manipulation.dss_to_tree("./lehigh.dss")

# add 2 15.6 kW wind turbines to each load
tree_turb = wiires.dss_manipulation.add_turbine(tree, 2, '15.6') 

# add a monitor at each load and substation 
tree_turb_mon = wiires.dss_manipulation.add_monitor(tree_turb)

# remove duplicate objects from tree (export statements, monitors, etc.)
tree_turb_mon_cleaned = wiires.dss_manipulation.remove_dups(tree_turb_mon)

# convert tree object back to dss and save to local directory 
wiires.dss_manipulation.tree_to_dss(tree_turb_mon_cleaned, "./mod_lehigh.dss")

# run dss file and save one csv per monitor locally (csv charts voltages, currents, and angles)
wiires.graph_dss.run_dss("./mod_lehigh.dss") 

# graph a three phase voltage chart 
wiires.graph_dss.graph_three_phase("./lehigh_Mon_load.671_command_center.csv")

# graph a single phase voltage chart 
wiires.graph_dss.graph_single_phase("./lehigh_Mon_load.634a_data_center.csv")

# get Copernicus ERA5 weather data for any coordinate for any year 1950-2019 (takes several hours)
weather_ds = wiires.LCEM.get_weather(39.952437, -75.16378, 2019)

# get hourly solar panel output given weather data for a particular year
solar_output_ds = wiires.LCEM.get_solar(weather_ds)

# get hourly wind turbine output given weather data for a particular year
wind_output_ds = wiires.LCEM.get_wind(weather_ds)

# find the lowest cost energy mix using grid search for a year of hourly load shapes given a location and year
# function accepts loads (csv or list), max solar, wind, and storage capacity in Watts, latitude, longitude, year, and step size for grid search
# solar, wind, and grid electricity costs as well as demand charge are preset but can be changed in code

opt_mix = wiires.LCEM.optimal_mix("./all_loads_vertical.csv", 0, 60_000, 0, 60_000, 0, 60_000, 39.952437, -75.16378, 2019, 5_000)

# get the data frame of storage/curtailed generation/renewables/fossil/demand/wind/solar levels hourly for a particular preset of solar/wind/storage capacity for a particular location and year, graph the mix, get the total cost of the system, and get the total wattage of grid electricity used in the year 
df, chart, total_cost, fossil_total = wiires.LCEM.calc_ren_mix("./all_loads_vertical.csv", 10_000, 10_000, 55_000, 39.952437, -75.16378, 2019)
# print(df[0:10])
# print(total_cost)
# print(fossil_total)

# a refined grid search gets the lowest cost energy mix down to the nearest Watt of capacity 
ref_mix = wiires.LCEM.refined_LCEM('./all_loads_vertical.csv', 0, 60_000, 0, 60_000, 0, 60_000, 39.952437, -75.16378, 2019, 10_000)

# use SciPy optimization to find the lowest cost energy mix 
weather_ds = wiires.LCEM.get_weather(39.952437, -75.16378, 2019)
solar_output_ds = wiires.LCEM.get_solar(weather_ds)
wind_output_ds = wiires.LCEM.get_wind(weather_ds)
results = wiires.OEM("./all_loads_vertical.csv", solar_output_ds, wind_output_ds)

# use hosting_cap.py to find the hosting capacity of any OpenDSS circuit
# hosting_cap.py adds 15.6 kW turbines to each load in the circuit incrementally until a load reaches 1.05 times the nominal voltage
# hosting_cap.py prints the bus that hit hosting capacity, prints the amount of generation needed to push the bus to hosting capacity, and outputs a plot of the circuit to networkPlot.png with the buses over hosting capacity outlined in red
# get_hosting_cap() takes 3 arguments: the circuit name, the starting count of turbines, and the stopping count of turbines. If the circuit does not reach hosting capacity within the range, the program will notify the user by print statement.

wiires.hosting_cap.get_hosting_cap("lehigh.dss", 170, 200)