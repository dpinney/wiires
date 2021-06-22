''' 
Adds a custom count of Bergey Excel 15 wind turbines to each bus of an OpenDSS circuit.
Includes functions to transform circuit between string and tree formats. 
Includes function to add monitors at each load of an OpenDSS tree.
Includes function to remove duplicate elements from an OpenDSS tree. 
'''

import opendssdirect as dss
import fire
import copy
import numpy as np

def dss_to_tree(path_to_dss):
  ''' Convert a .dss file to an in-memory, OMF-compatible 'tree' object.
	Note that we only support a VERY specifically-formatted DSS file.'''
	# TODO: Consider removing the handling for 'wdg=' syntax within this block, as we will not support it in an input file. 
	# Ingest file.
  with open(path_to_dss, 'r') as dss_file:
    contents = dss_file.readlines()
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
					# TODO: Should we pull the multiwinding transformer handling out of here and put it into dss_filePrep()?
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
	#with open('dssTreeRepresentation.csv', 'w') as out_file:
	#	ii = 1
	#	for k,v in contents.items():
	#		out_file.write(str(k) + '\n')
	#		ii = ii + 1
	#		for k2,v2 in v.items():
	#			out_file.write(',' + str(k2) + ',' + str(v2) + '\n')
  return list(contents.values())


def tree_to_dss(tree_object, output_path):
  out_file = open(output_path, 'w')
  for ob in tree_object:
    line = ob['!CMD']
    for key in ob:
      if not key.startswith('!'):
        line = f"{line} {key}={ob[key]}"
      if key.startswith('!TEST'):
        line = f"{line} {ob['!TEST']}"
        print(line)
    out_file.write(line + '\n')
  out_file.close()


def add_turbine(dss_tree, turb_count, kw):
  tree_copy = copy.deepcopy(dss_tree)
  # get names of all buses
  load_buses = [y.get('bus1') for y in tree_copy if y.get('object','').startswith('load.')]
  # get only the buses that have at least one load  
  load_buses = list(dict.fromkeys(load_buses))
  # add a wind turbine at each load bus immediately before solve statement 
  for i in load_buses:
    for s in range(turb_count):
      tree_copy.insert(tree_copy.index([x for x in tree_copy if x.get('!CMD','').startswith('solve')][0]), {'!CMD': 'new',
	    'object': f'generator.wind_{i}_{s}',
	    'bus': f'{i}.1.2.3',
	    'kw': kw,
	    'pf': '1.0',
	    'conn': 'delta',
	    'model': '1'})
  return tree_copy


def add_monitor(dss_tree):
  tree_copy = copy.deepcopy(dss_tree)
  # get names of all loads	
  loads = [y.get('object') for y in tree_copy if y.get('object','').startswith('load.')]
  # add a monitor at each load immediately before solve statement
  for i in loads:
	  tree_copy.insert(tree_copy.index([x for x in tree_copy if x.get('!CMD','').startswith('solve')][0]), {'!CMD': 'new',
	  'object': f'monitor.{i}',
	  'element': i,
	  'terminal': '1'})
  # get names of all substations
  substations = [x.get('object') for x in tree_copy if x.get('object','').startswith('vsource')]
  # add a monitor at each substation immediately before solve statement
  for i in substations:
    tree_copy.insert(tree_copy.index([x for x in tree_copy if x.get('!CMD','').startswith('solve')][0]), {'!CMD': 'new',
	  'object': f'monitor.{i}',
	  'element': i,
	  'terminal': '1',
	  'mode': '1'})		
    # add an export statement for each monitor 
    export_list = substations + loads
    for i in export_list:
      tree_copy.insert(tree_copy.index([x for x in tree_copy if x.get('!CMD','').startswith('export')][0]), {'!CMD': 'export ' f'monitors {i}'})
  return tree_copy


def remove_dups(dss_tree):
	# remove duplicate dictionaries 
	seen = set()
	new_l = []
	for d in dss_tree:
	    tup = tuple(d.items())
	    if tup not in seen:
	        seen.add(tup)
	        new_l.append(d)
	return new_l


def host_cap_dss_arrange(dss_tree, hour):
  tree_copy = copy.deepcopy(dss_tree)
  # all loadshape names 
  loadshapes = [y.get('object') for y in tree_copy if y.get('object','').startswith('loadshape.')]
  # list of loadshape names that are only for loads
  load_loadshape_names = ['loadshape.' + y.get('daily') for y in tree_copy if y.get('object','').startswith('load.')]
  # list of loadshape names that aren't for loads
  gen_loadshape_names = np.setdiff1d(loadshapes,load_loadshape_names)

  if hour == None:
    # replace first value of all load loadshapes with the minimum of those loadshapes  
    for i in load_loadshape_names:
      for x in tree_copy:
        if x.get('object','').startswith(i):
          mults_string = x.get('mult') 
          mults = [float(i) for i in mults_string[1:-1].split(',')]
          low_load = min(mults)
          mults[0] = low_load
          new_mults_string = str(mults)
          new_mults_string = new_mults_string.replace(" ","")
          x.update({'mult' : new_mults_string})
    # replace first value of all generation loadshapes with the maximum of those loadshapes 
    for i in gen_loadshape_names:
      for x in tree_copy:
        if x.get('object','').startswith(i):
          mults_string = x.get('mult') 
          mults = [float(i) for i in mults_string[1:-1].split(',')]
          high_gen = max(mults)
          mults[0] = high_gen
          new_mults_string = str(mults)
          new_mults_string = new_mults_string.replace(" ","")
          x.update({'mult' : new_mults_string})

  if hour != None:
    for i in loadshapes:
      for x in tree_copy:
        if x.get('object','').startswith(i):
          mults_string = x.get('mult') 
          mults = [float(i) for i in mults_string[1:-1].split(',')]
          hour_load = mults[hour - 1]
          mults[0] = hour_load
          new_mults_string = str(mults)
          new_mults_string = new_mults_string.replace(" ","")
          x.update({'mult' : new_mults_string})
  return tree_copy


# tree = dss_to_tree("data/wto_buses_xy.dss")
# import pprint as pp
# pp.pprint(tree)
# tree_to_dss(tree, "data/test_circuit.dss")