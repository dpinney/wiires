REopt_replacement install instructions (todo: make this into a script)

Download Julia v. 1.9.3

Start Julia in terminal (may need to add download location to your PATH):
> julia

Enter package manager:
> ]

Install the following packages:
> add REopt
> add JuMP
> add JSON
> add SCIP

Additionally packages (may not be necessary depending on solver decision):
> add Cbc