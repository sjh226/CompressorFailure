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
		SELECT WW.WellFlac
			  ,WW.Asset
			  ,WW.WellName
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

	df = df.dropna(axis=0, how='any', subset=['WellFlac', 'make_model'])

	return df

def comp_link(df):
	# Data on all compressors
	comps = pd.read_csv('data/compressors.csv', encoding = 'ISO-8859-1')
	comps['make_model'] = comps['Compressor Manufacturer'].str.lower() + ' ' + comps['Compressor Model'].str.lower()
	comps['WellName'] = comps['Well Name'].str.upper()
	comps['WellName'] = comps['WellName'].str.replace('/', '_')
	comps_lim = comps[['WellName', 'Meter', 'make_model']].dropna(how='all')

	# Merge surface failures with detailed compressor data
	joined = pd.merge(df, comps_lim, on='WellName', how='outer')

	return joined

def compressor_plot(df):
	plt.close()

	fig, ax = plt.subplots(1, 1, figsize=(20, 15))

	percent_dic = {}
	for compressor in df['make_model'].unique():
		percent_dic[compressor] = df[(df['make_model'] == compressor) & (df['fail_count'] > 0)].shape[0] \
								  / df[df['make_model'] == compressor].shape[0]

	comp_fail_dic = {}
	comp_tot_dic = {}
	for compressor in sorted(percent_dic, key=percent_dic.__getitem__):
		comp_fail_dic[compressor] = df[(df['make_model'] == compressor) & (df['fail_count'] > 0)].shape[0]
		comp_tot_dic[compressor] = df[df['make_model'] == compressor].shape[0]



	ind = np.arange(len(df['make_model'].unique()))
	width = 0.35

	p1 = plt.bar(ind, comp_tot_dic.values(), width, color='#5A1000')
	p2 = plt.bar(ind, comp_fail_dic.values(), width, color='#E12800')

	plt.ylabel('Total Compressors')
	plt.title('Compressor Failures')
	plt.xlabel('Compressor Type')
	plt.xticks(ind, comp_tot_dic.keys(), rotation='vertical')
	plt.legend((p1[0], p2[0]), ('Total Installs', 'Compressors which Failed in 2017'))

	plt.savefig('comp_fails.png')


if __name__ == '__main__':
	df = failures_fetch()
	compressor_plot(df)
