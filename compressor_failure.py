import pandas as pd
import numpy as np
import pyodbc
import sys
import re
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from failures import


def rtr_fetch(well_flac):
	try:
		connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-L48.BP.Com;Database=EDW;trusted_connection=yes')
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		SELECT DDH.DateTime
			  ,DDH.Well1_Asset
			  ,DDH.Well1_WellFlac AS WellFlac
			  ,DDH.Well1_WellName AS WellName
			  ,DDH.Well1_CasingPress
			  ,DDH.Well1_TubingPress
			  ,DDH.RTU1_Barometer
			  ,DDH.RTU1_BatteryVoltage
			  ,DDH.Meter1_ID
			  ,DDH.Meter1_StaticPressPDayAvg
			  ,DDH.Meter1_Temperature
		FROM [EDW].[RTR].[DataDailyHistory] AS DDH\
		WHERE DDH.Well1_WellFlac = '""" + str(well_flac) +"""'
		ORDER BY DDH.DateTime ASC;
	""")
	### How do we insert different values for WellFlac?

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None

	failure_df = failures_fetch(well_flac)
	# print(failure_df['fail_date'])

	def last_date(date):
		early_dates = sorted(failure_df[failure_df['fail_date'] <= date]['fail_date'].values)
		if early_dates:
			return early_dates[-1]
		else:
			return np.nan

	df['last_failure'] = df['DateTime'].apply(last_date)

	return df


def failures_fetch(well_flac):
	try:
		connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-L48.BP.Com;Database=EDW;trusted_connection=yes')
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()

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

	SQLCommand = ("""
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
		   AND W.WellFlac = '"""+ str(well_flac) +"""'
		GROUP BY W.WellFlac, SF.surfaceFailureDate, W.Asset, W.ID, S.Comp1_Status, DW.WellName
		ORDER BY W.WellFlac, fail_date;
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None

	df['fail_date'] = pd.to_datetime(df['fail_date'])

	return df

def time_series_model():
	# Do we want to use RTR data here?
	# Maybe look at production data, or calculate an average from RTR for each day.

	pass


if __name__ == '__main__':
	df = rtr_fetch(70075401)
