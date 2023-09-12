import json, time
from os.path import join as pJoin
import wiires, julia
from julia import Main

#for graphs
import matplotlib.pyplot as plt
import numpy as np

########################################################
#functions for converting REopt input to REopt.jl input
########################################################

#dictionary mapping REopt variable name to REopt.jl variable name 
#  plus necessary information on parent section & data type
def init_reopt_to_julia_dict():

# ( translated variable name , section(s)(/none) , datatype(None if multiple types) )
    to_jl_dict = { "Site":("Site", {None}, dict),
              "latitude":("latitude",{"Site"},float),
              "longitude":("longitude",{"Site"},float),

              "ElectricTariff":("ElectricTariff", {None}, dict), 
              "wholesale_rate_us_dollars_per_kwh":("wholesale_rate", {"ElectricTariff"}, None),
              "blended_annual_rates_us_dollars_per_kwh":
              ("blended_annual_energy_rate", {"ElectricTariff"}, None),
              "blended_annual_demand_charges_us_dollars_per_kw":
              ("blended_annual_demand_rate", {"ElectricTariff"}, None),

              "LoadProfile":("ElectricLoad", {None}, dict),
              "critical_load_pct":("critical_load_fraction",{"ElectricLoad"}, float),
              "loads_kw":("loads_kw",{"ElectricLoad"},list), "year":("year",{"ElectricLoad"},int), 
              "loads_kw_is_net":("loads_kw_is_net",{"ElectricLoad"},bool),

              "Financial":("Financial", {None}, dict),
              "value_of_lost_load_us_dollars_per_kwh":("value_of_lost_load_per_kwh", {"Financial"}, float), 
              "om_cost_escalation_pct":("om_cost_escalation_rate_fraction", {"Financial"}, float),
              "offtaker_discount_pct":("offtaker_discount_rate_fraction",{"Financial"}, float),
              "analysis_years":("analysis_years",{"Financial"}, int),

              #PV, ElectricStorage, Wind & Generator shared variables:
              "installed_cost_us_dollars_per_kw":
              ("installed_cost_per_kw", {"PV","Wind","ElectricStorage","Generator"} ,float),
              "min_kw":("min_kw", {"PV","Wind","ElectricStorage","Generator"}, float),
              "max_kw":("max_kw", {"PV","Wind","ElectricStorage","Generator"}, float), 
              "macrs_option_years":("macrs_option_years", {"PV","Wind","ElectricStorage","Generator"}, int),

              #PV, Wind, Generatior shared:
              "can_export_beyond_site_load":("can_export_beyond_nem_limit", {"PV","Wind","Generator"}, bool),
              "federal_itc_pct":("federal_itc_fraction", {"PV","Wind","Generator"}, float),

              #Generator & ElectricStorage shared:
              "replace_cost_us_dollars_per_kw":("replace_cost_per_kw",{"Generator","ElectricStorage"},float),

              #PV & Generator shared:
              "can_curtail":("can_curtail", {"PV", "Generator"}, bool), 
              "existing_kw":("existing_kw", {"PV", "Generator"}, float), 
              "om_cost_us_dollars_per_kw":("om_cost_per_kw", {"PV", "Generator"}, float),

              "PV":("PV", {None}, dict),

              "Storage":("ElectricStorage", {None}, dict),
              "replace_cost_us_dollars_per_kwh":("replace_cost_per_kwh", {"ElectricStorage"}, float),
              "total_itc_percent":("total_itc_fraction", {"ElectricStorage"}, float),
              "inverter_replacement_year":("inverter_replacement_year", {"ElectricStorage"}, int),
              "battery_replacement_year":("battery_replacement_year", {"ElectricStorage"}, int), 
              "min_kwh":("min_kwh", {"ElectricStorage"}, float), 
              "max_kwh":("max_kwh", {"ElectricStorage"}, float),
              "installed_cost_us_dollars_per_kwh":("installed_cost_per_kwh", {"ElectricStorage"}, float),

              "Wind":("Wind", {None}, dict),

              "Generator":("Generator", {None}, dict),
              "generator_only_runs_during_grid_outage":("only_runs_during_grid_outage",{"Generator"}, bool),
              "fuel_avail_gal":("fuel_avail_gal", {"Generator"}, float), 
              "min_turn_down_pct":("min_turn_down_fraction", {"Generator"}, float),
              "diesel_fuel_cost_us_dollars_per_gallon":("fuel_cost_per_gallon", {"Generator"}, float),
              "emissions_factor_lb_CO2_per_gal":("emissions_factor_lb_CO2_per_gal", {"Generator"}, float),
              "om_cost_us_dollars_per_kwh":("om_cost_per_kwh", {"Generator"}, float),

              #### ElectricUtility (not used in REopt)
              "outage_start_time_step":("outage_start_time_step", {"ElectricUtility"}, int), 
              "outage_end_time_step":("outage_end_time_step", {"ElectricUtility"}, int)
    }

    #variables in reopt that aren't used in reopt.jl
    not_included_in_jl = { "outage_is_major_event", "wholesale_rate_above_site_load_us_dollars_per_kwh" }

    return (to_jl_dict, not_included_in_jl)

#checks if variable name is used in REopt.jl
def check_key(key, to_jl_dict, not_included_in_jl):
    if key in not_included_in_jl:
        return False
    elif key not in to_jl_dict:
        print("error: json key not found: " + str(key))
        return False
    else:
        return True
    
#returns data value if it is the correct data type or converts if feasible 
def check_input(key,var,to_jl_dict):
    if key in to_jl_dict:
        var_type = to_jl_dict[key][2]
        if var_type == type(var) or var_type == None:
            return var
        elif var_type == int:
            return int(var)
        elif var_type == float:
            return float(var)
    else:
        print("error key not found: " + str(key))
        return None

#returns converted section name if used in REopt.jl
def get_section(key,section,to_jl_dict):
    section_to_jl = to_jl_dict[section][0]
    new_sections = to_jl_dict[key][1]
    if section_to_jl in new_sections:
        return section_to_jl
    elif len(new_sections) == 1:
        return list(new_sections)[0]
    else:
        print("error: no sections found for: " + str(key))
        return None

#converts variable into REopt.jl version of variable and adds to json
def add_variable(section,key,value,jl_json,to_jl_dict):
    new_section = get_section(key,section,to_jl_dict)
    new_var_name = to_jl_dict[key][0]
    if not new_section in jl_json:
        jl_json[new_section] = dict()
    jl_json[new_section][new_var_name] = check_input(key,value,to_jl_dict)
    return jl_json


#converts an input json for REopt to the equivalent version for REopt.jl

#REopt json: {Scenario: { Site: {lat, lon, ElectricTariff:{}, LoadProfile:{}, etc. }}}
#REopt.jl json: { Site: { lat, lon }, ElectricTariff:{}, LoadProfile:{}, etc. }
def convert_to_jl(reopt_json):
    (to_jl_dict,not_included_in_jl) = init_reopt_to_julia_dict()
    new_jl_json = {}
    scenario = reopt_json["Scenario"]["Site"] #todo: add error checking
    for key in scenario:
        if not check_key(key,to_jl_dict,not_included_in_jl):
            return 
        value = scenario[key]
        if not type(value) == dict:
            new_jl_json = add_variable("Site",key,value,new_jl_json,to_jl_dict)
        else:
            for sub_key in value:
                if not check_key(sub_key,to_jl_dict,not_included_in_jl): 
                    if sub_key in not_included_in_jl:
                        continue
                    else:
                        return 
                else:
                    new_jl_json = add_variable(key,sub_key,value[sub_key],new_jl_json,to_jl_dict)
    return new_jl_json

################################
#functions for running REopt.jl
################################

def get_json_file(path):
    test_json = {}
    with open(path) as jsonFile:
        test_json = json.load(jsonFile)
    return test_json

#default values used in REopt in microgridDesign
def init_default_julia_json():

    #default REopt.jl input json
    julia_scenario = { #"Scenario": {
        "Site": {
            "latitude": 39.7817,
            "longitude": -89.6501
        },
        "ElectricTariff": {
            "wholesale_rate": 0.034, #"wholesale_rate_us_dollars_per_kwh": wholesaleCost,
            #"wholesale_rate_above_site_load_us_dollars_per_kwh": wholesaleCost
            #"urdb_label": '5b75cfe95457a3454faf0aea',
            #since urdb_label is 'off':
            "blended_annual_energy_rate": 0.1,
            "blended_annual_demand_rate": 20
        },
        "ElectricLoad": { #"LoadProfile": {
            #giving csv path instead of parsing out values from csv (only option in REopt.jl)
            "path_to_csv": "testFiles/loadShape.csv",
            #"loads_kw": jsonifiableLoad,
            "year": 2017
        },
        "Financial": {
            "value_of_lost_load_per_kwh": 100.0, #"value_of_lost_load_us_dollars_per_kwh": value_of_lost_load,
            "analysis_years": 25,
            "om_cost_escalation_rate_fraction": 0.025, #"om_cost_escalation_pct": omCostEscalator,
            "offtaker_discount_rate_fraction": 0.083 #"offtaker_discount_pct": discountRate
            #^includes owner_discount_rate_fraction as well?
        },
        "PV": {
            "installed_cost_per_kw": 1600, #installed_cost_us_dollars_per_kw": solarCost,
            "min_kw": 0,
            "can_export_beyond_nem_limit": True, #"can_export_beyond_site_load": solarCanExport,
            "can_curtail": True,
            "macrs_option_years": 5,
            "federal_itc_fraction": 0.26 # "federal_itc_pct": solarItcPercent
        },
        "ElectricStorage": { #"Storage": {
            "installed_cost_per_kw": 840, #"installed_cost_us_dollars_per_kw": batteryPowerCost,
            "installed_cost_per_kwh": 420, #"installed_cost_us_dollars_per_kwh": batteryCapacityCost,
            "replace_cost_per_kw": 410, #"replace_cost_us_dollars_per_kw": batteryPowerCostReplace,
            "replace_cost_per_kwh": 200, #"replace_cost_us_dollars_per_kwh": batteryCapacityCostReplace,
            "inverter_replacement_year": 10,
            "battery_replacement_year": 10,
            "min_kw": 0,
            "min_kwh": 0,
            "macrs_option_years": 7,
            "total_itc_fraction": 0 #"total_itc_percent": batteryItcPercent
        },
        "Wind": {
            "installed_cost_per_kw": 4898,#"installed_cost_us_dollars_per_kw": windCost,
            "min_kw": 0,
            "macrs_option_years": 5,
            "federal_itc_fraction": 0.26 #"federal_itc_pct": windItcPercent
        },
        "Generator": {
            "installed_cost_per_kw": 500,#"installed_cost_us_dollars_per_kw": dieselGenCost,
            "only_runs_during_grid_outage": True, #"generator_only_runs_during_grid_outage": dieselOnlyRunsDuringOutage,
            "macrs_option_years": 0
        }
    }

    return julia_scenario

#runs julia scenario in REopt.jl (found in ./test.jl)
def reopt_jl_test(scenario, filePath, outPath):
    with open(filePath, "w") as jsonFile:
        json.dump(scenario, jsonFile)

    julia_code = f"""
	include("test.jl")
	main("{filePath}","{outPath}")
	"""

    Main.eval(julia_code)

##########################################
# functions for displaying REopt.jl output
##########################################

#displays graph given output data from REopt.jl
def make_graph(x,ys,xlabel,ylabel,title):
    plt.figure(figsize=(10, 6))
    for (y,label) in ys:
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


#displays results of REopt.jl call (options to display in terminal or through html)
def display_julia_output_json(filepath, charts="terminal"):
    #reopt_julia_output_json = {} 
    #with open(filepath) as jsonFile:
    #    reopt_julia_output_json = json.load(jsonFile)
    reopt_julia_output_json = get_json_file(filepath)

    #todo: account for some sections being optional
    financial = reopt_julia_output_json['Financial']
    electricTariff = reopt_julia_output_json['ElectricTariff']
    electricLoad = reopt_julia_output_json['ElectricLoad']
    electricUtility = reopt_julia_output_json['ElectricUtility']
    pv = reopt_julia_output_json['PV']
    wind = reopt_julia_output_json['Wind']
    electricStorage = reopt_julia_output_json['ElectricStorage']
    generator = reopt_julia_output_json['Generator']

    if charts == "terminal":
        #design overview
        print("Total solar: " + str(pv['size_kw']))
        print("Total wind: " + str(wind['size_kw']))
        print("Total inverter: " + str(electricStorage['size_kw']))
        print("Total storage: " + str(electricStorage['size_kwh']))
        print("Total fossil: " + str(generator['size_kw']))

        #load overview
        batt_to_load = (electricStorage['storage_to_load_series_kw'], "Battery to Load")
        pv_to_load = (pv['electric_to_load_series_kw'], "PV to Load")
        grid_to_load = (electricUtility['electric_to_load_series_kw'], "Grid to Load")
        wind_to_load = (wind['electric_to_load_series_kw'], "Wind to Load")
        diesel_to_load = (generator['electric_to_load_series_kw'], "Generator to Load")
        ys = [batt_to_load, pv_to_load, grid_to_load, wind_to_load, diesel_to_load]

        x = np.arange(8760) #8760 = hours in a year

        make_graph(x, ys,"time","Power (kW)","Load Overview")

        #solar generation
        #wind generation
        #fossil generation
        #battery charge source
        #battery charge percentage

    #todo: present basic overview of the data in html
    elif charts == "html":
        return 

###########################################################################
#goal: compare outcomes of REopt, REopt.jl, WIIRES on the same test cases 
###########################################################################

if __name__ == "__main__":
    start_time = time.time()
    
    #test_filename = "Scenario_test_POST.json"
    #test_file_path = "/Users/lilyolson/Documents/reopt crash examples/CE Test Case/" + test_filename
    #reopt_test_json = get_json_file(test_file_path)

    #test_json_to_jl = convert_to_jl(reopt_test_json)
    
    #test_converted_input = "Scenario_test_POST_CE_Test_case.json"
    #test_outfile = "output_CE.json"
    #reopt_jl_test(test_json_to_jl,test_converted_input, test_outfile)
    

    julia_scenario = init_default_julia_json()
    path = "/Users/lilyolson/Documents/nreca/wiires/REopt_replacements/testFiles/"
    test_filepath = path + "Scenario_test_POST_julia.json"
    test_outfile = path + "output.json"
    reopt_jl_test(julia_scenario, test_filepath, test_outfile)
    display_julia_output_json(test_outfile) 

    end_time = time.time()
    print("time taken")
    print(end_time - start_time)