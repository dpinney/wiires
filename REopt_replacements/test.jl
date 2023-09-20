using REopt, JuMP, Cbc, JSON, SCIP


function get_model(solver::String)
	if solver == "SCIP"
		m = Model(SCIP.Optimizer)
		#testing SCIP attributes
		#set_attribute(m, "display/verblevel", 0) # default is 4 - only change once done testing
		set_attribute(m, "limits/gap", 0.08) #default is 0
		set_attribute(m, "limits/solutions", 20) #default is infinity
		set_attribute(m, "lp/threads", 8) #default = 0, max = 64
		set_attribute(m, "parallel/minnthreads", 4) # default = 1, max = 64 
		return m

	elseif solver == "Cbc"
		m = Model(Cbc.Optimizer)
		#testing how these impact runtime (for Cbc)
		#set_attribute(m,"threads",4)
		#set_attribute(m,"maxSolutions",1)
		#set_attribute(m,"maxNodes",100)
		#set_attribute(m,"ratioGap",0.05)
		return m
	else 
		println("Error: invalid solver")
	end
end


function results_to_json(results, output_path)
	j = JSON.json(results)

	output_file = output_path * ".json"
	open(output_file, "w") do file
		println(file, j)
	end
end


#available solvers: SCIP, Cbc
#microgrid_only => not used in OMF REopt? only used in simulate_outages
function main(json_path::String, output_path::String, solver::String="SCIP", 
	outages::Bool=false, microgrid_only::Bool=false)

	m = get_model(solver)

	results = Dict()
	json_file = json_path * ".json"

	#note: only running BAUScenario 
	m2 = get_model(solver)
	results = run_reopt([m,m2],json_file)

	results_to_json(results, output_path)

	if outages
		reopt_inputs = REoptInputs(json_file)
		outage_results = simulate_outages(results, reopt_inputs; microgrid_only=microgrid_only)

		outage_path = output_path * "_outages"
		results_to_json(outage_results,outage_path)
	end

end