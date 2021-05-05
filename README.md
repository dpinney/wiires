# wiires
Wind Integration Into Rural Energy Systems

# Installation

`pip install git+https://github.com/dpinney/wiires`

# Usage Examples
```
>>> import wiires

>>> # convert a dss file to an object for manipulation 
>>> tree = wiires.dssmanipulation.dssToTree('lehigh.dss')
>>> tree

>>> # add 2 15.6 kW wind turbines to each load
>>> tree_turb = wiires.dssmanipulation.addTurbine(tree, 2, '15.6') 
>>> tree_turb

>>> 
```
