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
import random
import math
import feeder
import json

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

  # replace first loadshapes with loadshapes of the hour at which the timeseries hosting capacity is reached 
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


def dssToOmd(dssFilePath, RADIUS=0.0002):
  ''' Generate an OMD.
  SIDE-EFFECTS: creates the OMD'''
  # Injecting additional coordinates.
  #TODO: derive sensible RADIUS from lat/lon numbers.
  tree = dss_to_tree(dssFilePath)
  evil_glm = evilDssTreeToGldTree(tree)
  name_map = _name_to_key(evil_glm)
  for ob in evil_glm.values():
    ob_name = ob.get('name','')
    ob_type = ob.get('object','')
    if 'parent' in ob:
      parent_loc = name_map[ob['parent']]
      parent_ob = evil_glm[parent_loc]
      parent_lat = parent_ob.get('latitude', None)
      parent_lon = parent_ob.get('longitude', None)
      # place randomly on circle around parent.
      angle = random.random()*3.14159265*2;
      x = math.cos(angle)*RADIUS;
      y = math.sin(angle)*RADIUS;
      ob['latitude'] = str(float(parent_lat) + x)
      ob['longitude'] = str(float(parent_lon) + y)
      # print(ob)
  return evil_glm


def evilDssTreeToGldTree(dssTree):
  ''' World's worst and ugliest converter. Hence evil. 
  We built this to do quick-and-dirty viz of openDSS files. '''
  gldTree = {}
  g_id = 1
  # Build bad gld representation of each object
  bus_names = []
  bus_with_coords = []
  # Handle all the components.
  for ob in dssTree:
    try:
      if ob['!CMD'] == 'setbusxy':
        gldTree[str(g_id)] = {
          'object': 'bus',
          'name': ob['bus'],
          'latitude': ob['y'],
          'longitude': ob['x']
        }
        bus_with_coords.append(ob['bus'])
      elif ob['!CMD'] == 'new':
        obtype, name = ob['object'].split('.', maxsplit=1)
        if 'bus1' in ob and 'bus2' in ob:
          # line-like object. includes reactors.
          fro, froCode = ob['bus1'].split('.', maxsplit=1)
          to, toCode = ob['bus2'].split('.', maxsplit=1)
          gldTree[str(g_id)] = {
            'object': obtype,
            'name': name,
            'from': fro,
            'to': to,
            '!FROCODE': '.' + froCode,
            '!TOCODE': '.' + toCode
          }
          bus_names.extend([fro, to])
          stuff = gldTree[str(g_id)]
          _extend_with_exc(ob, stuff, ['object','bus1','bus2','!CMD'])
        elif 'buses' in ob:
          #transformer-like object.
          bb = ob['buses']
          bb = bb.replace(']','').replace('[','').split(',')
          b1 = bb[0]
          fro, froCode = b1.split('.', maxsplit=1)
          ob['!FROCODE'] = '.' + froCode
          b2 = bb[1]
          to, toCode = b2.split('.', maxsplit=1)
          ob['!TOCODE'] = '.' + toCode
          gldobj = {
            'object': obtype,
            'name': name,
            'from': fro,
            'to': to
          }
          bus_names.extend([fro, to])
          if len(bb)==3:
            b3 = bb[2]
            to2, to2Code = b3.split('.', maxsplit=1)
            ob['!TO2CODE'] = '.' + to2Code
            gldobj['to2'] = to2
            bus_names.append(to2)
          gldTree[str(g_id)] = gldobj
          _extend_with_exc(ob, gldTree[str(g_id)], ['object','buses','!CMD'])
        elif 'bus' in ob:
          #load-like object.
          bus_root, connCode = ob['bus'].split('.', maxsplit=1)
          gldTree[str(g_id)] = {
            'object': obtype,
            'name': name,
            'parent': bus_root,
            '!CONNCODE': '.' + connCode
          }
          bus_names.append(bus_root)
          _extend_with_exc(ob, gldTree[str(g_id)], ['object','bus','!CMD'])
        elif 'bus1' in ob and 'bus2' not in ob:
          #load-like object, alternate syntax
          try:
            bus_root, connCode = ob['bus1'].split('.', maxsplit=1)
            ob['!CONNCODE'] = '.' + connCode
          except:
            bus_root = ob['bus1'] # this shoudln't happen if the .clean syntax guide is followed.
          gldTree[str(g_id)] = {
            'object': obtype,
            'name': name,
            'parent': bus_root,
          }
          bus_names.append(bus_root)
          _extend_with_exc(ob, gldTree[str(g_id)], ['object','bus1','!CMD'])
        elif 'element' in ob:
          #control object (connected to another object instead of a bus)
          #cobtype, cobname, connCode = ob['element'].split('.', maxsplit=2)
          cobtype, cobname = ob['element'].split('.', maxsplit=1)
          gldTree[str(g_id)] = {
            'object': obtype,
            'name': name,
            'parent': cobtype + '.' + cobname,
          }
          _extend_with_exc(ob, gldTree[str(g_id)], ['object','element','!CMD'])
        else:
          #config-like object
          gldTree[str(g_id)] = {
            'object': obtype,
            'name': name
          }
          _extend_with_exc(ob, gldTree[str(g_id)], ['object','!CMD'])
      elif ob.get('object','').split('.')[0]=='vsource':
        obtype, name = ob['object'].split('.')
        conn, connCode = ob.get('bus1').split('.', maxsplit=1)
        gldTree[str(g_id)] = {
          'object': obtype,
          'name': name,
          'parent': conn,
          '!CONNCODE': '.' + connCode
        }
        _extend_with_exc(ob, gldTree[str(g_id)], ['object','bus1'])
      elif ob['!CMD']=='edit':
        #TODO: handle edited objects? maybe just extend the 'new' block (excluding vsource) because the functionality is basically the same.
        warnings.warn(f"Ignored 'edit' command: {ob}")
      elif ob['!CMD'] not in ['new', 'setbusxy', 'edit']: # what about 'set', 
        #command-like objects.
        gldTree[str(g_id)] = {
          'object': '!CMD',
          'name': ob['!CMD']
        }
        _extend_with_exc(ob, gldTree[str(g_id)], ['!CMD'])
      else:
        warnings.warn(f"Ignored {ob}")
      g_id += 1
    except:
      raise Exception(f"\n\nError encountered on parsing object {ob}\n")
  # Warn on buses with no coords.
  #no_coord_buses = set(bus_names) - set(bus_with_coords)
  #if len(no_coord_buses) != 0:
    #warnings.warn(f"Buses without coordinates:{no_coord_buses}")
  return gldTree


def _extend_with_exc(from_d, to_d, exclude_list):
  ''' Add all items in from_d to to_d that aren't in exclude_list. '''
  good_items = {k: from_d[k] for k in from_d if k not in exclude_list}
  to_d.update(good_items)


def _name_to_key(glm):
  ''' Make fast lookup map by name in a glm.
  WARNING: if the glm changes, the map will no longer be valid.'''
  mapping = {}
  for key, val in glm.items():
    if 'name' in val:
      mapping[val['name']] = key
  return mapping


def evilToOmd(evilTree, outPath):
  omdStruct = dict(feeder.newFeederWireframe)
  omdStruct['syntax'] = 'DSS'
  omdStruct['tree'] = evilTree
  with open(outPath, 'w') as outFile:
    json.dump(omdStruct, outFile, indent=4)
