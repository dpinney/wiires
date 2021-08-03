import os
import sys, shutil, csv, base64
from datetime import datetime as dt, timedelta
import pulp
from os.path import isdir, join as pJoin
from numpy import npv
import pandas as pd

def work(modelDir, inputDict):
	''' Model processing done here. '''
	# dispatchStrategy = str(inputDict.get('dispatchStrategy'))
	# if dispatchStrategy == 'prediction':
	# 	return forecastWork(modelDir, inputDict)

	out = {}  # See bottom of file for out's structure
	cellCapacity, dischargeRate, chargeRate, cellQuantity, demandCharge, cellCost, retailCost = \
		[float(inputDict[x]) for x in ('cellCapacity', 'dischargeRate', 'chargeRate',
			'cellQuantity', 'demandCharge', 'cellCost', 'retailCost')]

	projYears, batteryCycleLife = [int(inputDict[x]) for x in ('projYears', 'batteryCycleLife')]
	
	discountRate = float(inputDict.get('discountRate')) / 100.0
	dodFactor = float(inputDict.get('dodFactor')) / 100.0

	# Efficiency calculation temporarily removed
	inverterEfficiency = float(inputDict.get('inverterEfficiency')) / 100.0
	# Note: inverterEfficiency is squared to get round trip efficiency.
	battEff = float(inputDict.get('batteryEfficiency')) / 100.0 * (inverterEfficiency ** 2)

	demand = csvValidateAndLoad(
		inputDict['demandCurve'], 
		modelDir, header=None, 
		ignore_nans=False,
		save_file='demand.csv'
	)[0]

	dates = [(dt(2019, 1, 1) + timedelta(hours=1)*x) for x in range(8760)]
	dc = [{'power': load, 'month': date.month -1, 'hour': date.hour} for load, date in zip(demand, dates)]

	# if dispatchStrategy == 'customDispatch':
	# 	customDispatch = csvValidateAndLoad(
	# 		inputDict['customDispatchStrategy'], 
	# 		modelDir, header=None, 
	# 		ignore_nans=False,
	# 		save_file='dispatchStrategy.csv'
	# 	)[0]
	# 	for c, d in zip(customDispatch, dc):
	# 		d['dispatch'] = c
	# 	assert all(['dispatch' in d for d in dc])  # ensure each row is filled

	# list of 12 lists of monthly demands
	demandByMonth = [[t['power'] for t in dc if t['month']==x] for x in range(12)]
	monthlyPeakDemand = [max(lDemands) for lDemands in demandByMonth]
	battCapacity = cellQuantity * cellCapacity * dodFactor
	battDischarge = cellQuantity * dischargeRate 
	battCharge = cellQuantity * chargeRate 

	SoC = battCapacity 
	# if dispatchStrategy == 'optimal':
	ps = [battDischarge] * 12
	# keep shrinking peak shave (ps) until every month doesn't fully expend the battery
	while True:
		SoC = battCapacity # (700kW)
		incorrect_shave = [False] * 12 
		for row in dc:
			month = row['month']
			if not incorrect_shave[month]:
				powerUnderPeak = monthlyPeakDemand[month] - row['power'] - ps[month] 
				charge = (min(powerUnderPeak, battCharge, battCapacity - SoC) if powerUnderPeak > 0 
					else -1 * min(abs(powerUnderPeak), battDischarge, SoC))
				# print(powerUnderPeak, charge, SoC)
				if charge == -1 * SoC: 
					incorrect_shave[month] = True
				SoC += charge 
				# SoC = 0 when incorrect_shave[month] == True 
				row['netpower'] = row['power'] + charge 
				row['battSoC'] = SoC
		ps = [s-1 if incorrect else s for s, incorrect in zip(ps, incorrect_shave)]
		if not any(incorrect_shave):
			break
	# elif dispatchStrategy == 'daily':
	# 	start = int(inputDict.get('startPeakHour'))
	# 	end = int(inputDict.get('endPeakHour'))
	# 	for r in dc:
	# 		# Discharge if hour is within peak hours otherwise charge
	# 		charge = (-1*min(battDischarge, SoC) if start <= r['hour'] <= end 
	# 			else min(battCharge, battCapacity - SoC))
	# 		r['netpower'] = r['power'] + charge
	# 		SoC += charge
	# 		r['battSoC'] = SoC
	# elif dispatchStrategy == 'customDispatch':
	# 	for r in dc:
	# 		# Discharge if there is a 1 in the dispatch strategy csv, otherwise charge the battery.
	# 		charge = (-1*min(battDischarge, SoC) if r['dispatch'] == 1 
	# 			else min(battCharge, battCapacity-SoC))
	# 		r['netpower'] = r['power'] + charge
	# 		SoC += charge
	# 		r['battSoC'] = SoC

	# ------------------------- CALCULATIONS ------------------------- #
	netByMonth = [[t['netpower'] for t in dc if t['month']==x] for x in range(12)]
	monthlyPeakNet = [max(net) for net in netByMonth]
	ps = [h-s for h, s in zip(monthlyPeakDemand, monthlyPeakNet)]
	# Monthly Cost Comparison Table
	out['monthlyDemand'] = [sum(lDemand)/1000 for lDemand in demandByMonth]
	out['monthlyDemandRed'] = [t-p for t, p in zip(out['monthlyDemand'], ps)]
	out['ps'] = ps
	out['benefitMonthly'] = [x*demandCharge for x in ps]
	
	# Demand Before and After Storage Graph
	out['demand'] = [t['power']*1000.0 for t in dc] # kW -> W
	out['demandAfterBattery'] = [t['netpower']*1000.0 for t in dc] # kW -> W
	out['batteryDischargekW'] = [d-b for d, b in zip(out['demand'], out['demandAfterBattery'])]
	out['batteryDischargekWMax'] = max(out['batteryDischargekW'])

	with open(pJoin(modelDir, 'batteryDispatch.txt'), 'w') as f:
		f.write('\n'.join([str(x) for x in out['batteryDischargekW']]) + '\n')

	# Battery State of Charge Graph
	out['batterySoc'] = SoC = [t['battSoC']/battCapacity*100*dodFactor + (100-100*dodFactor) for t in dc]
	# Estimate number of cyles the battery went through. Sums the percent of SoC.
	cycleEquivalents = sum([SoC[i]-SoC[i+1] for i, x in enumerate(SoC[:-1]) if SoC[i+1] < SoC[i]]) / 100.0
	out['cycleEquivalents'] = cycleEquivalents
	out['batteryLife'] = batteryCycleLife / cycleEquivalents

	# Cash Flow Graph
	cashFlowCurve = [sum(ps)*demandCharge for year in range(projYears)]
	cashFlowCurve.insert(0, -1 * cellCost * cellQuantity)  # insert initial investment
	# simplePayback is also affected by the cost to recharge the battery every day of the year
	out['SPP'] = (cellCost*cellQuantity)/(sum(ps)*demandCharge)
	out['netCashflow'] = cashFlowCurve
	out['cumulativeCashflow'] = [sum(cashFlowCurve[:i+1]) for i, d in enumerate(cashFlowCurve)]
	out['NPV'] = npv(discountRate, cashFlowCurve)

	battCostPerCycle = cellQuantity * cellCost / batteryCycleLife
	lcoeTotCost = cycleEquivalents*retailCost + battCostPerCycle*cycleEquivalents
	out['LCOE'] = lcoeTotCost / (cycleEquivalents*battCapacity)

	# Other
	out['startDate'] = '2019-01-01'  # dc[0]['datetime'].isoformat()
	out['stderr'] = ''
	# Seemingly unimportant. Ask permission to delete.
	out['stdout'] = 'Success' 

	return out


def safe_assert(bool_statement, error_str, keep_running):
	if keep_running:
		if not bool_statement:
			print(error_str)
	else:
		assert bool_statement, error_str


def csvValidateAndLoad(file_input, modelDir, header=0, nrows=8760, ncols=1, dtypes=[], return_type='list_by_col', ignore_nans=False, save_file=None, ignore_errors=False):
	"""
		Safely validates, loads, and saves user's file input for model's use.
		Parameters:
		file_input: stream from input_dict to be read
		modelDir: a temporary or permanent file saved to given location
		header: row of header, enter "None" if no header provided.
		nrows: skip confirmation if None
		ncols: skip confirmation if None
		dtypes: dtypes as columns should be parsed. If empty, no parsing. 
						Use "False" for column index where there should be no parsing.
					 	This can be used as any mapping function.
		return_type: options: 'dict', 'df', 'list_by_col', 'list_by_row'
		ignore_nans: Ignore NaN values
		save_file: if not None, save file with given *relative* pathname. It will be appended to modelDir
		ignore_errors (bool): if True, allow program to keep running when errors found and print
		Returns:
		Datatype as dictated by input parameters
	"""

	# save temporary file
	temp_path = os.path.join(modelDir, 'csv_temp.csv') if save_file == None else os.path.join(modelDir, save_file)
	with open(temp_path, 'w') as f:
		f.write(file_input)
	df = pd.read_csv(temp_path, header=header)
	
	if nrows != None:
		safe_assert( df.shape[0] == nrows, (
			f'Incorrect CSV size. Required: {nrows} rows. Given: {df.shape[0]} rows.'
		), ignore_errors)

	if ncols != None:
		safe_assert( df.shape[1] == ncols, (
			f'Incorrect CSV size. Required: {ncols} columns. Given: {df.shape[1]} columns.'
		), ignore_errors)
	
	# NaNs
	if not ignore_nans:
		d = df.isna().any().to_dict()
		nan_columns = [k for k, v in d.items() if v]
		safe_assert( 
			len(nan_columns) == 0, 
			f'NaNs detected in columns {nan_columns}. Please adjust your CSV accordingly.',
			ignore_errors
		)
	
	# parse datatypes
	safe_assert(
		(len(dtypes) == 0) or (len(dtypes) == ncols), 
		f"Length of dtypes parser must match ncols, you've entered {len(dtypes)}. If no parsing, provide empty array.",
		ignore_errors
	)
	for t, x in zip(dtypes, df.columns):
		if t != False:
			df[x] = df[x].map(lambda x: t(x))
	
	# delete file if requested
	if save_file == None:
		os.remove(temp_path)

	# return proper type
	OPTIONS = ['dict', 'df', 'list_by_col', 'list_by_row']
	safe_assert(
		return_type in OPTIONS, 
		f'return_type not recognized. Options are {OPTIONS}.',
		ignore_errors
	)
	if return_type == 'list_by_col':
		return [df[x].tolist() for x in df.columns]
	elif return_type == 'list_by_row':
		return df.values.tolist()
	elif return_type == 'df':
		return df
	elif return_type == 'dict':
		return [{k: v for k, v in row.items()} for _, row in df.iterrows()]



# INPUT VARIABLES
_myDir = os.path.dirname(os.path.abspath(__file__))
_omfDir = os.path.dirname(_myDir)
with open(pJoin(_omfDir,'wiires','data','all_loads_vertical.csv')) as f:
	demand_curve = f.read()
inputDict = {'cellCapacity':350,'dischargeRate':250,'chargeRate':250,'cellQuantity':100,'demandCharge':20,'cellCost':7140,'retailCost':0.06,'projYears':15,'batteryCycleLife':5000,'discountRate':2.5,'dodFactor':100,'inverterEfficiency':100,'batteryEfficiency':100,'demandCurve':demand_curve}
modelDir = './data/'
test = work(modelDir, inputDict)
# print(test)

