import json, time
import os
from os.path import join as pJoin
import wiires, julia
from julia import Main

#for graphs
import matplotlib.pyplot as plt
import numpy as np

#for adding interactive graphs to html
import mpld3

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
              "urdb_label":("urdb_label",{"ElectricTariff"},str),

              "LoadProfile":("ElectricLoad", {None}, dict),
              "critical_loads_kw":("critical_loads_kw",{"ElectricLoad"},list),
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

              #PV, Wind, Generator shared:
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
    test_json = None
    file = path + ".json"
    if os.path.exists(file):
        with open(file) as jsonFile:
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
def reopt_jl_test(scenario, filePath, outPath, solver="SCIP", outages=False, microgrid_only=False):
    file = filePath + ".json"
    with open(file, "w") as jsonFile:
        json.dump(scenario, jsonFile)

    Main.include("test.jl")
    Main.main(filePath, outPath, solver, outages, microgrid_only)

##########################################
# functions for displaying REopt.jl output
##########################################

#generates html for microgrid overview given REopt.jl output json and resilience output json
def microgrid_overview_table(testName, json, outage_json=None):
    retStr = f'''<p>
    Recommended Microgrid Design Overview for {testName} <br>
    <table border=1px cellspacing=0>
    '''

    totalSavings = json.get('Financial',{}).get("npv",0)
    load = json.get('ElectricLoad',{}).get("load_series_kw",0)
    avgLoad = round(sum(load)/len(load),1)

    totalSolar = json.get('PV',{}).get('size_kw',0)
    totalWind = json.get('Wind',{}).get('size_kw',0)
    totalInverter = json.get('ElectricStorage',{}).get('size_kw',0)
    totalStorage = json.get('ElectricStorage',{}).get('size_kwh',0)
    totalFossil = json.get('Generator',{}).get('size_kw',0)

    avgOutage = 0
    if outage_json:
        avgOutage = outage_json["resilience_hours_avg"]

    #is this equivalent to REopt value?
    dieselUsed = json.get('Generator',{}).get('annual_fuel_consumption_gal',0)
    # generator_fuel_used_per_outage_gal? check source code

    retStr += f'''
      <tr>
            <th> Total Savings </th>
            <th> Average Load (kWh) </th>
            <th> Total Solar (kW) </th>
            <th> Total Wind (kW) </th>
            <th> Total Inverter (kW) </th>
            <th> Total Storage (kWh) </th>
            <th> Total Fossil (kW) </th>
            <th> Average length of survived Outage (hours) </th>
            <th> Fossil Fuel consumed during specified Outage (diesel gal equiv) </th>
            <th>  </th>
        </tr>
        <tr>
            <td> {totalSavings} </td>
            <td> {avgLoad} </td>
            <td> {totalSolar} </td>
            <td> {totalWind} </td>
            <td> {totalInverter} </td>
            <td> {totalStorage} </td>
            <td> {totalFossil} </td>
            <td> {avgOutage} </td>
            <td> {dieselUsed} </td>
        </tr>
    </table>
    </p>
    '''
    return retStr

#generates html for financial performance given REopt.jl output json
def financial_performance_table(testName, json):
    retStr = f''' <p>
    Microgrid Lifetime Financial Performance for {testName} <br>
    <table border=1px cellspacing=0>
    '''
    h = [ ["", "Business as Usual", "Microgrid", "Difference"],
         ["Demand Cost", "", "", ""],
         ["Energy Cost", "", "", ""],
         ["Total Cost", "", "", ""] ]
    
    et = json.get(('ElectricTariff'),{})
    f = json.get('Financial',{})

    #todo: round all to 2 decimals [ round(val, 2) ]
    h[1][1] = et.get("lifecycle_demand_cost_after_tax_bau")
    h[1][2] = et.get("lifecycle_demand_cost_after_tax")
    h[1][3] = h[1][1] - h[1][2]
    h[2][1] = et.get("lifecycle_demand_cost_after_tax_bau")
    h[2][2] = et.get("lifecycle_demand_cost_after_tax")
    h[2][3] = h[2][1] - h[2][2]
    h[3][1] = f.get("lcc_bau")
    h[3][2] = f.get("lcc")
    h[3][3] = h[3][1] - h[3][2]
    
    for i in range(4):
        retStr += "<tr>"
        for j in range(4):
            isHeader = i == 0 or j == 0
            retStr += "<th>" if isHeader else "<td>"
            retStr += str(h[i][j])
            retStr += "</th>" if isHeader else "</td>"
        retStr += "</tr>"

    retStr += "</table></p>"
    return retStr

#todo: generate html for proforma analysis given REopt.jl output json
def proforma_table(json):
    retStr = '''<p>
    <table border=1px cellspacing=0>
    '''
    retStr += "</table></p>"
    return retStr

#displays graph given output data from REopt.jl
def make_graph(x,ys,xlabel,ylabel,title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for (y,label) in ys:
        if label == "":
            ax.plot(x,y)
        else:
            ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.legend()
    plt.grid(True)
    return (fig, ax)

def all_graphs(json, outage_json):
    retStr = ""

    electricUtility = json.get('ElectricUtility',{})
    pv = json.get('PV',{})
    wind = json.get('Wind',{})
    electricStorage = json.get('ElectricStorage',{})
    generator = json.get('Generator',{})

    load_overview = []
    battery_charge_source = []
    
    x = np.arange(8760) #8760 = hours in a year

    if electricStorage:
        load_overview.append((electricStorage['storage_to_load_series_kw'], "Battery to Load"))
        battery_charge_percentage = [ (electricStorage['soc_series_fraction'],'') ]

        (fig_batt_percent,ax_batt_percent) = make_graph(x, battery_charge_percentage,"time","%", 
                                                  "Battery Charge Percentage")
        
    if pv:
        load_overview.append((pv['electric_to_load_series_kw'], "PV to Load"))
        battery_charge_source.append((pv['electric_to_storage_series_kw'], "Solar"))

        solar_generation = [(pv['electric_to_load_series_kw'], "PV used to meet Load")]
        solar_generation.append((pv['electric_curtailed_series_kw'], "PV Curtailed"))
        solar_generation.append((pv['electric_to_grid_series_kw'], "PV exported to Grid"))
        solar_generation.append((pv['electric_to_storage_series_kw'], "PV used to charge Battery"))

        (fig_pv,ax_pv) = make_graph(x, solar_generation,"time","Power (kW)","Solar Generation")
        retStr += mpld3.fig_to_html(fig_pv)

    if electricUtility:
        load_overview.append((electricUtility['electric_to_load_series_kw'], "Grid to Load"))
        battery_charge_source.append((electricUtility['electric_to_storage_series_kw'], "Grid"))

    if wind:
        load_overview.append((wind['electric_to_load_series_kw'], "Wind to Load"))
        battery_charge_source.append((wind['electric_to_storage_series_kw'], "Wind"))

        wind_generation = [ (wind['electric_to_load_series_kw'], "Wind used to meet Load") ]
        wind_generation.append((wind['electric_to_storage_series_kw'], "Wind used to charge Battery"))

        (fig_wind,ax_wind) = make_graph(x, wind_generation,"time","Power (kW)","Wind Generation")
        retStr += mpld3.fig_to_html(fig_wind)

    if generator:
        load_overview.append((generator['electric_to_load_series_kw'], "Generator to Load"))
        battery_charge_source.append((generator['electric_to_storage_series_kw'], "Fossil Gen"))
        
        fossil_generation = [ (generator['electric_to_load_series_kw'],"Fossil Gen used to meet Load") ]
        fossil_generation.append((generator['electric_to_storage_series_kw'],
                                  "Fossil Gen used to charge Battery"))

        (fig_fossil,ax_fossil) = make_graph(x, fossil_generation,"time","Power (kW)","Fossil Generation")
        retStr += mpld3.fig_to_html(fig_fossil)

    (fig_load,ax_load) = make_graph(x, load_overview,"time","Power (kW)","Load Overview")
    retStr = mpld3.fig_to_html(fig_load) + retStr

    if electricStorage:
        (fig_batt_source,ax_batt_source) = make_graph(x, battery_charge_source,"time","Power (kW)",
                                                  "Battery Charge Source")
        retStr += mpld3.fig_to_html(fig_batt_source)
        retStr += mpld3.fig_to_html(fig_batt_percent)

    if outage_json != None:

        resilience = outage_json["resilience_by_time_step"]
        res_y = [(resilience,"")]
        res_x = np.arange(len(resilience))
        (fig_res,ax_res) = make_graph(res_x, res_y, "Start Hour", "Longest Outage Survived (hours)",
                                      "Resilience Overview")
        retStr += mpld3.fig_to_html(fig_res)

        survival_prob_x = outage_json["outage_durations"]
        survival_prob_y = [(outage_json["probs_of_surviving"], "")]

        (fig_survival, ax_survival) = make_graph(survival_prob_x, survival_prob_y, "Outage Length (hours)",
                                                "Probability of Meeting Critical Load", 
                                                "Outage Survival Probability")
        retStr += mpld3.fig_to_html(fig_survival)

    return retStr


#displays results of REopt.jl call in html file
def display_julia_output_json(filepath, total_runtime, method="run_reopt"):
    output_json = get_json_file(filepath)
    outage_output = get_json_file(filepath + "_outages")

    html_graphs = all_graphs(output_json, outage_output)

    html_doc = f'''
    <body>
    <p>
    <b>Total runtime: {total_runtime} seconds</b>
    </p>
    {microgrid_overview_table(output_json,outage_output)}
    {financial_performance_table(output_json)}
    '''
    html_doc += html_graphs + "</body>"
    #to do: different output files based on test name
    Html_file= open("testFiles/sample_test.html","w")
    Html_file.write(html_doc)
    Html_file.close()


def get_test_overview(testPath, runtime, solver, simulate_outages):
    tab = "&nbsp;"
    retStr = "<p>"
    retStr += "Test File: " + testPath + "<br>"
    retStr += tab + "Runtime: " + str(runtime) + "<br>"
    retStr += tab + "Solver: " + solver + "<br>"
    retStr += tab + "Simulates outages? "
    retStr += "Yes <br></p>" if simulate_outages else "No <br></p>"
    return retStr


#for comparing mutiple test outputs
# tests = [ (testPath, runtime, solver, simulate_outages ), ... ]
def html_comparison(tests):
    html_doc = "<body>"
    overview_str = ""
    microgrid_overview_str = ""
    financial_performance_str = ""
    graphs = []

    for (testPath, testName, runtime, solver, simulate_outages) in tests:
        overview_str += get_test_overview(testPath,runtime,solver,simulate_outages)

        output_json = get_json_file(testPath)
        outage_json = get_json_file(testPath + "_outages")
        test = testName + " ( solver: " + solver + " )"

        microgrid_overview_str += microgrid_overview_table(test, output_json, outage_json)
        financial_performance_str += financial_performance_table(test, output_json)

        graph_set = all_graphs(output_json, outage_json)
        graphs.append((graph_set,test))

    html_doc += overview_str + microgrid_overview_str + financial_performance_str
    for (graph,test) in graphs:
        html_doc += "<p> graphs for " + test + "<br>"
        html_doc += graph + "</p>"

    html_doc += "</body>"
    html_file= open("testFiles/sample_comparison_test.html","w")
    html_file.write(html_doc)
    html_file.close()

###########################################################################
#goal: compare outcomes of REopt, REopt.jl, WIIRES on the same test cases 
###########################################################################

#gives paths for the converted input json and output json from REopt.jl
def getFilePaths(testName):
    #todo: make relative
    path = "/Users/lilyolson/Documents/nreca/wiires/REopt_replacements/testFiles/"
    #path to write converted input json to
    converted_input_path = path + "Scenario_test_" + testName #+ ".json"
    #path to write json output of REopt.jl to
    outPath = path + "output_" + testName #+ ".json"
    return (converted_input_path, outPath)

#todo: give option to convert input json for julia (ie: convert = True/False)
def runFullTest(path, testName, fileName, solver="SCIP", outages=True):
    start = time.time()

    #to add: if file already exists at outpath and 'get_cached' input = True
    # => return previous results (to save time when running a lot of test cases)
    filePath = path + fileName
    Json = get_json_file(filePath)
    jl_json = convert_to_jl(Json)
    (inPath, outPath) = getFilePaths(testName)
    reopt_jl_test(jl_json, inPath, outPath, solver=solver, outages=outages)

    end = time.time()
    runtime = end - start
    #to do: allow for non-SCIP in input
    return(outPath, testName, runtime, solver, outages)

if __name__ == "__main__":
    all_tests = []

    test_filename = "Scenario_test_POST" #.json"

    ########### CONWAY_MG_MAX
    CONWAY_path = "/Users/lilyolson/Documents/reopt crash examples/CONWAY_MG_MAX/"
    CONWAY_testName = "CONWAY_MG_MAX"
    CONWAY_filename = test_filename
    CONWAY_output = runFullTest(CONWAY_path, CONWAY_testName, CONWAY_filename)
    all_tests.append(CONWAY_output)

    CONWAY_Cbc_output = runFullTest(CONWAY_path, CONWAY_testName, CONWAY_filename, solver="Cbc")
    all_tests.append(CONWAY_Cbc_output)

    ############### CE test case
    #to do: give gap size as optional input to reopt_jl_test (or termination time)
    #CE_path = "/Users/lilyolson/Documents/reopt crash examples/CE Test Case/" + test_filename
    #CE_json = get_json_file(CE_path)
    #CE_jl_json = convert_to_jl(CE_json)

    ############## CONWAY_30MAY23_SOLARBATTERY
    CONWAY_SB_path = "/Users/lilyolson/Documents/reopt crash examples/CONWAY_30MAY23_SOLARBATTERY/"
    CONWAY_SB_testName = "CONWAY_30MAY23_SOLARBATTERY"
    CONWAY_SB_filename = test_filename
    CONWAY_SB_output = runFullTest(CONWAY_SB_path, CONWAY_SB_testName, CONWAY_SB_filename)
    all_tests.append(CONWAY_SB_output)

    CONWAY_SB_Cbc_output = runFullTest(CONWAY_SB_path, CONWAY_SB_testName, CONWAY_SB_filename, solver="Cbc")
    all_tests.append(CONWAY_SB_Cbc_output)

    ####### default julia json (from microgridDesign)
    default_start = time.time()

    julia_scenario = init_default_julia_json()
    (default_in, default_out) = getFilePaths("julia")
    reopt_jl_test(julia_scenario, default_in, default_out)
    default_end = time.time()
    default_runtime = default_end - default_start
    all_tests.append((default_out, "julia default", default_runtime, "SCIP", ""))

    default_Cbc_start = time.time()
    reopt_jl_test(julia_scenario, default_in, default_out, solver="Cbc")
    default_Cbc_end = time.time()
    default_Cbc_runtime = default_Cbc_end - default_Cbc_start
    all_tests.append((default_out, "julia default", default_runtime, "Cbc", ""))

    #display_julia_output_json(outputPath, total_runtime, method="simulate_outages")

    html_comparison(all_tests) # work in progress