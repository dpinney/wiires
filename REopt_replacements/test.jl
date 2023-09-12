using REopt, JuMP, Cbc, JSON

function main(json_path::String, output_path::String)
	m = Model(Cbc.Optimizer)

	#testing how these impact runtime
	set_attribute(m,"threads",4)
	#set_attribute(m,"maxSolutions",1)
	set_attribute(m,"maxNodes",100)
	set_attribute(m,"ratioGap",0.05)

	results = run_reopt(m,json_path)
	j = JSON.json(results)

	open(output_path, "w") do file
		println(file, j)
	end
end