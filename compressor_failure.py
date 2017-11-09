import pandas as pd
import numpy as np
import pyodbc
import sys
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from failures import comp_link, failure_classifier


def rtr_fetch(well_flac):
	try:
		connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-L48.BP.Com;Database=EDW;trusted_connection=yes')
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	# Fetch the daily RTR history for the specific well
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

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None

	# Query for the surface failures on record for a specific well
	failure_df = failures_fetch(well_flac)

	# Function to be applied to current day to calculate the last fail date
	def last_date(date):
		early_dates = sorted(failure_df[failure_df['fail_date'] <= date]['fail_date'].values)
		if early_dates:
			return early_dates[-1]
		else:
			return np.nan

	df['last_failure'] = df['DateTime'].apply(last_date)
	df['failure'] = np.where(df['last_failure'] == df['DateTime'], 1, 0)
	df['days_since_fail'] = df['DateTime'] - df['last_failure']

	# Need this to be dynamic
	# Use SQL query with the compresser csv
	fail_df = comp_link()

	# Bring in the make and model of compressor for this well along with the
	# total percentage of these compressors that fail
	df['comp_model'] = fail_df[fail_df['WellFlac'] == well_flac]['make_model'].values[0]

	# Should this be calculated for the current specific date?
	# Maybe pass model into a RF each time to use stacked model
	df['percent_failure'] = fail_df[fail_df['WellFlac'] == well_flac]['fail_percentage'].values[0]
	return df


def failures_fetch(well_flac):
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

	# Ensure dates are in the correct format
	df['fail_date'] = pd.to_datetime(df['fail_date'])

	return df

def time_series_model(df):
	# Do we want to use RTR data here?
	# Maybe look at production data, or calculate an average from RTR for each day.
	def fail_in(date):
		days = 7
		if np.mean(df[(df['DateTime'] > date) & \
					  (df['DateTime'] <= date + \
					  datetime.timedelta(days=days))]['failure']) > 0:
			return 1
		else:
			return 0

	df['fail_in_week'] = df['DateTime'].apply(fail_in)

	# Build and return RF model based solely on make and model
	fail_df = comp_link()
	model_pred = failure_classifier(fail_df, model=df['comp_model'].unique()[0], results=False)

	df['model_prediction'] = model_pred[0]

	df['days_since_fail'] = pd.to_numeric(df['days_since_fail'])

	test_date = df.iloc[int(df.shape[0] * .7),:]['DateTime']
	train = df[df['DateTime'] < test_date]
	test = df[df['DateTime'] >= test_date]

	train = train.drop(['DateTime', 'Well1_Asset', 'WellFlac', 'WellName', 'comp_model', 'last_failure'], axis=1)
	test = test.drop(['DateTime', 'Well1_Asset', 'WellFlac', 'WellName', 'comp_model', 'last_failure'], axis=1)

	y_train = train.pop('fail_in_week')
	y_test = test.pop('fail_in_week')

	rf = RandomForestClassifier()
	rf.fit(train, y_train)
	accuracy = rf.score(test, y_test)

	print('Accuracy of last RF model:\n{}'.format(accuracy))

	return df


if __name__ == '__main__':
	df = rtr_fetch(70075401)

	df = time_series_model(df)
