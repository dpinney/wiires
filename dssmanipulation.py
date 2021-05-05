''' 
Adds a custom count of Bergey Excel 15 wind turbines to each bus of an OpenDSS circuit.
Includes functions to transform circuit between string and tree formats. 
Includes function to add monitors at each load of an OpenDSS tree.
Includes function to remove duplicate elements from an OpenDSS tree. 
'''

# import os
# import glob
# import xarray
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import opendssdirect as dss
import fire
import pickle
import copy

def dssToTree(pathToDss):
  ''' Convert a .dss file to an in-memory, OMF-compatible 'tree' object.
	Note that we only support a VERY specifically-formatted DSS file.'''
	# TODO: Consider removing the handling for 'wdg=' syntax within this block, as we will not support it in an input file. 
	# Ingest file.
  with open(pathToDss, 'r') as dssFile:
    contents = dssFile.readlines()
	# Lowercase everything. OpenDSS is case insensitive.
  contents = [x.lower() for x in contents]
	# Clean up the file.
  for i, line in enumerate(contents):
    # Remove whitespace.
    contents[i] = line.strip()
		# Comment removal
    bangLoc = line.find('!')
    if bangLoc != -1: 
      contents[i] = line[:bangLoc]
    # Join using the tilde (~) syntax
    if line.startswith('~'):
      # Look back to find the first line with content.
      for j in range(i - 1, 0, -1):
        if contents[j] != '':
          contents[j] = contents[j] + contents[i].replace('~', ' ')
          contents[i] = ''
          break
	# Capture original line numbers and drop blanks
  contents = dict([(c,x) for (c, x) in enumerate(contents) if x != ''])
	# Lex it
  convTbl = {'bus':'buses', 'conn':'conns', 'kv':'kvs', 'kva':'kvas', '%r':'%r'}
  # convTbl = {'bus':'buses', 'conn':'conns', 'kv':'kvs', 'kva':'kvas', '%r':'%rs'} # TODO at some point this will need to happen; need to check what is affected i.e. viz, etc

  from collections import OrderedDict 
  for i, line in contents.items():
    jpos = 0
    if line.startswith('export'):
      contents[i] = OrderedDict({'!CMD':'export ' + line[7:]})
      continue
    try:
      contents[i] = line.split()
      ob = OrderedDict() 
      ob['!CMD'] = contents[i][0]
      if len(contents[i]) > 1:
        for j in range(1, len(contents[i])):
          jpos = j
          splitlen = len(contents[i][j].split('='))
          k,v=('','',)
        #   print(contents[i], splitlen)
          if splitlen==3:
            print('OMF does not support OpenDSS\'s \'file=\' syntax for defining property values.')
            k,v,f = contents[i][j].split('=')
						# replaceFileSyntax() # DEBUG
						## replaceFileSyntax  should do the following:
						# parse the filename (contained in v)
						# read in the file and parse as array
						# v = file content array, cast as a string
          else:
            k,v = contents[i][j].split('=')
					# TODO: Should we pull the multiwinding transformer handling out of here and put it into dssFilePrep()?
          if k == 'wdg':
            continue
          if (k in ob.keys()) or (convTbl.get(k,k) in ob.keys()): # if the single key already exists in the object, then this is the second pass. If pluralized key exists, then this is the 2+nth pass
            # pluralize the key if needed, get the existing values, add the incoming value, place into ob, remove singular key
            plurlk = convTbl.get(k, None) # use conversion table to pluralize the key or keep same key
            incmngVal = v
            xistngVals = []
            if k in ob: # indicates 2nd winding, existing value is a string (in the case of %r, this indicates 3rd winding as well!)
              if (type(ob[k]) != tuple) or (type(ob[k]) != list): # pluralized values can be defined as either
							#if iter(type(ob[k])):
                xistngVals.append(ob[k])
                del ob[k]
            if plurlk in ob: # indicates 3rd+ winding; existing values are tuples
              for item in ob[plurlk]:
                xistngVals.append(item)
            xistngVals.append(incmngVal) # concatenate incoming value with the existing values
            ob[plurlk] =  tuple(xistngVals) # convert all to tuple
          else: # if single key has not already been added, add it
            ob[k] = v
    except:
      raise Exception(f"Error encountered in group (space delimited) #{jpos+1} of line {i + 1}: {line}")
			# raise Exception("Temp fix but error in loop at line 76")
    if type(ob) is OrderedDict:
      contents[i] = ob
	# Print to file
	#with open('dssTreeRepresentation.csv', 'w') as outFile:
	#	ii = 1
	#	for k,v in contents.items():
	#		outFile.write(str(k) + '\n')
	#		ii = ii + 1
	#		for k2,v2 in v.items():
	#			outFile.write(',' + str(k2) + ',' + str(v2) + '\n')
  return list(contents.values())


def treeToDss(treeObject, outputPath):
  outFile = open(outputPath, 'w')
  for ob in treeObject:
    line = ob['!CMD']
    for key in ob:
      if not key.startswith('!'):
        line = f"{line} {key}={ob[key]}"
      if key.startswith('!TEST'):
        line = f"{line} {ob['!TEST']}"
        print(line)
    outFile.write(line + '\n')
  outFile.close()


def addTurbine(dssTree, turbCount, kva):
  treeCopy = copy.deepcopy(dssTree)
  # get names of all buses
  buses = [x.get('bus') for x in treeCopy if x.get('!CMD','').startswith('setbusxy')]
  # add a wind turbine at each bus immediately before solve statement 
  for i in buses:
    for s in range(turbCount):
      treeCopy.insert(treeCopy.index([x for x in treeCopy if x.get('object','').startswith('monitor.')][0]), {'!CMD': 'new',
	    'object': f'generator.wind_{i}_{s}',
	    'bus': f'{i}.1.2.3',
	    'kva': kva,
	    'pf': '1.0',
	    'conn': 'delta',
	    'duty': 'wind',
	    'model': '1'})
  return treeCopy


def addMonitor(dssTree):
  treeCopy = copy.deepcopy(dssTree)
  # get names of all loads	
  loads = [y.get('object') for y in treeCopy if y.get('object','').startswith('load.')]
  # add a monitor at each load immediately before solve statement
  for i in loads:
	  treeCopy.insert(t.index([x for x in treeCopy if x.get('!CMD','').startswith('solve')][0]), {'!CMD': 'new',
	  'object': f'monitor.{i}',
	  'element': i,
	  'terminal': '1'})
  # get names of all substations
  substations = [x.get('object') for x in treeCopy if x.get('object','').startswith('vsource')]
  # add a monitor at each substation immediately before solve statement
  for i in substations:
    treeCopy.insert(t.index([x for x in treeCopy if x.get('!CMD','').startswith('solve')][0]), {'!CMD': 'new',
	  'object': f'monitor.{i}',
	  'element': i,
	  'terminal': '1',
	  'mode': '1'})		
    # add an export statement for each monitor 
    exportList = substations + loads
    for i in exportList:
      treeCopy.insert(treeCopy.index([x for x in treeCopy if x.get('!CMD','').startswith('export')][0]), {'!CMD': 'export ' f'monitors {i}'})
  return treeCopy


def removeDups(dssTree):
	# remove duplicate dictionaries 
	seen = set()
	new_l = []
	for d in dssTree:
	    tup = tuple(d.items())
	    if tup not in seen:
	        seen.add(tup)
	        new_l.append(d)
	return new_l