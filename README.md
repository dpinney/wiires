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

>>> # add 2 15.6 kW wind turbines to each load
>>> tree_turb = wiires.dss_manipulation.add_turbine(tree, 2, '15.6') 
>>> tree_turb

>>> # add a monitor at each load and substation 
>>> tree_turb_mon = wiires.dss_manipulation.add_monitor(tree_turb)
>>> tree_turb_mon

>>> # remove duplicate objects from tree (export statements, monitors, etc.)
>>> tree_turb_mon_cleaned = wiires.dss_manipulation.remove_dups(tree_turb_mon)
>>> tree_turb_mon_cleaned

>>> # convert tree object back to dss and save to local directory 
>>> wiires.dss_manipulation.tree_to_dss(tree_turb_mon_cleaned, "./my_circuit.dss")

>>> # run dss file and save one csv per monitor locally (csv charts voltages, currents, and angles)
>>> wiires.graph_dss.run_dss("./my_circuit.dss") 

>>> # graph a three phase voltage chart 
>>> wiires.graph_dss.graph_three_phase("./monitor1_output.csv")

>>> # graph a single phase voltage chart 
>>> wiires.graph_dss.graph_single_phase("./monitor2_output.csv")

>>> # find the lowest cost energy mix using grid search for a year of hourly load shapes given a location and year
>>> # function accepts loads (csv or list), max solar, wind, and storage capacity in Watts, latitude, longitude, year, and step size for grid search
>>> # solar, wind, and grid electricity costs as well as demand charge are preset but can be changed in code
>>> LCEM = wiires.LCEM.optimal_mix("./loads.csv", 60_000_000, 60_000_000, 60_000_000, 39.952437, -75.16378, 2019, 5_000_000)
>>> LCEM

>>> # get the data frame of storage/curtailed generation/renewables/fossil/demand/wind/solar levels hourly for a particular preset of solar/wind/storage capacity for a particular location and year, graph the mix, get the total cost of the system, and get the total wattage of grid electricity used in the year 
>>> df, chart, total_cost, fossil_total = wiires.LCEM.calc_ren_mix("./loads.csv", 10_000_000, 10_000_000, 55_000_000, 39.952437, -75.16378, 2019)
>>> df
>>> chart.show()
>>> total_cost
>>> fossil_total
```
