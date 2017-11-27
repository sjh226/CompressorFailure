import pandas as pd
import numpy as np
import sys
import pyodbc
import matplotlib.pyplot as plt

def failures_fetch():
	try:
		connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-L48.BP.Com;Database=EDW;trusted_connection=yes')
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()

	# Create temporary table for compressor failures
	# Currently focused on Farmington
	SQLCommand = ("""
		USE OperationsDataMart

		DROP TABLE IF EXISTS #SurfaceFailures

		SELECT *
		INTO   #SurfaceFailures
		FROM   EDW.Enbase.SurfaceFailureActionDetailed AS SFAD
		WHERE  SFAD.assetBU IN ('Farmington')
		   AND SFAD.surfaceFailureComponent LIKE '%Compressor%'
		   AND SFAD.assetWellFlac IS NOT NULL;
	""")

	cursor.execute(SQLCommand)

	# Retrieve RTR data for a specific well and join surface failure information
	SQLCommand = ("""
		SELECT WW.WellFlac AS well_flac
			  ,WW.Asset AS asset
			  ,WW.WellName AS well_name
			  ,MAX(SFW.fail_date) AS last_fail
			  ,ISNULL(SUM(SFW.fail_count), 0) AS fail_count
		FROM [OperationsDataMart].[Dimensions].[Wells] AS WW
		LEFT OUTER JOIN(
			SELECT W.WellFlac
				  ,DW.WellName
				  ,W.Asset
				  ,W.ID
				  ,S.Comp1_Status
				  ,SF.surfaceFailureDate AS fail_date
				  ,COUNT(SF.surfaceFailureDate) AS fail_count
			FROM   EDW.RTR.ConfigWell W
				INNER JOIN
				EDW.RTR.DataCurrentSnapshot AS S
				ON W.ID = S.ID
				LEFT OUTER JOIN
				OperationsDataMart.Dimensions.Wells AS DW
				ON W.WellFlac = DW.WellFlac
				LEFT OUTER JOIN
				#SurfaceFailures AS SF
				ON SF.assetWellFlac = W.WellFlac
			WHERE  W.WellStatus = 'A'
			   AND S.Comp1_Status != 'NA'
			   AND W.Asset IN ('SJS')
			   AND W.WellFlac IS NOT NULL
			GROUP BY W.WellFlac, SF.surfaceFailureDate, W.Asset, W.ID, S.Comp1_Status, DW.WellName) AS SFW
		ON WW.WellFlac = SFW.WellFlac
		WHERE WW.Asset IN ('Farmington')
		GROUP BY WW.WellFlac, WW.Asset, WW.WellName;
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]

		# Ensure dates are in the correct format
		df['fail_date'] = pd.to_datetime(df['fail_date'])
	except:
		pass

	df = comp_link(df)

	df = df.dropna(axis=0, how='any', subset=['well_flac', 'make_model'])

	return df

def comp_link(df):
	# Data on all compressors
	comps = pd.read_csv('data/compressors.csv', encoding = 'ISO-8859-1')
	comps.columns = [col.lower().replace(' ', '_') for col in comps.columns]
	comps['make_model'] = comps['compressor_manufacturer'].str.lower() + ' ' + comps['compressor_model'].str.lower()
	comps['well_name'] = comps['well_name'].str.replace('/', '_')
	comps['rate'].replace('WELL OWNED', '0', inplace=True)
	comps['rate'].replace('3,750', '3750', inplace=True)
	comps['cost'] = comps['rate'].astype(float) + comps['maintenance_including_fluids'].astype(float)
	comps_lim = comps.dropna(how='all')

	# Merge surface failures with detailed compressor data
	joined = pd.merge(df, comps_lim, on='well_name', how='outer')

	return joined

def month_plot(df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	df.loc[df['last_fail'].notnull(), 'month'] = pd.to_datetime(df[df['last_fail'].notnull()]['last_fail']).dt.month
	month_fail = {}
	for month in np.arange(1, 13):
		month_fail[month] = df[df['month'] == month].shape[0]

	ind = np.arange(12)
	width = 0.35

	plt.bar(ind, month_fail.values(), width)
	plt.ylabel('Total Failures')
	plt.title('Compressor Failure by Month')
	plt.xlabel('Month')
	plt.xticks(ind, ('January', 'February', 'March', 'April', 'May', 'June', \
					 'July', 'August', 'September', 'October', 'November', 'December'), \
					 rotation='vertical')

	plt.savefig('images/monthly_fails.png')

def compressor_plot(df):
	plt.close()

	fig, ax = plt.subplots(1, 1, figsize=(20, 15))

	percent_dic = {}
	for compressor in df['make_model'].unique():
		percent_dic[compressor] = df[(df['make_model'] == compressor) & (df['fail_count'] > 0)].shape[0] \
								  / df[df['make_model'] == compressor].shape[0]

	comp_fail_dic = {}
	comp_tot_dic = {}
	comp_cost = {}
	for compressor in sorted(percent_dic, key=percent_dic.__getitem__):
		comp_fail_dic[compressor] = df[(df['make_model'] == compressor) & (df['fail_count'] > 0)].shape[0]
		comp_tot_dic[compressor] = df[df['make_model'] == compressor].shape[0]
		comp_cost[compressor] = df[df['make_model'] == compressor]['cost'].unique()[0]

	ind = np.arange(len(df['make_model'].unique()))
	width = 0.35

	p1 = plt.bar(ind, comp_tot_dic.values(), width, color='#5A1000')
	p2 = plt.bar(ind, comp_fail_dic.values(), width, color='#E12800')
	p3 = plt.bar(ind + width, comp_cost.values(), width)

	plt.ylabel('Total Compressors')
	plt.title('Compressor Failures')
	plt.xlabel('Compressor Type')
	plt.xticks(ind, comp_tot_dic.keys(), rotation='vertical')
	plt.legend((p1[0], p2[0], p3[0]), ('Total Installs', 'Compressors which Failed in 2017', 'Monthly Maintenance Cost'))

	plt.savefig('images/comp_fails.png')

def price_plot(df):
	plt.close()

	fig, ax1 = plt.subplots(1, 1, figsize=(20, 15))

	comp_list = []
	for compressor in df['make_model'].unique():
		if df[df['make_model'] == compressor].shape[0] > 10:
			comp_list.append(compressor)

	percent_dic = {}
	for compressor in comp_list:
		percent_dic[compressor] = df[(df['make_model'] == compressor) & (df['fail_count'] > 0)].shape[0] \
								  / df[df['make_model'] == compressor].shape[0]

	cost_dic = {}
	for compressor in comp_list:
		cost_dic[compressor] = df[df['make_model'] == compressor]['cost'].unique()[0]

	ind = np.arange(len(comp_list))
	width = 0.35

	p1 = ax1.bar(ind, percent_dic.values(), width, color='#58b5ef')
	ax1.set_ylabel('Percent Failure')
	ax1.set_xlabel('Compressor Type')
	plt.xticks(ind, comp_list, rotation='vertical')

	ax2 = ax1.twinx()
	p2 = ax2.bar(ind + width, cost_dic.values(), width, color='#39702b')
	ax2.set_ylabel('Compressor Cost')

	plt.title('Compressor Failures')
	plt.legend((p1[0], p2[0]), ('Percent Failure', 'Monthly Maintenance Cost'))

	plt.savefig('images/comp_cost.png')


if __name__ == '__main__':
	df = failures_fetch()
	# compressor_plot(df)
	# month_plot(df)
	price_plot(df)
