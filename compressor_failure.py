import pandas as pd
import numpy as np
import pyodbc
import sys
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


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
			  ,DDH.Well1_Asset AS Asset
			  ,DDH.Well1_WellFlac AS WellFlac
			  ,DDH.Well1_WellName AS WellName
			  ,DDH.Well1_CasingPress
			  ,DDH.Well1_TubingPress
			  ,DDH.RTU1_Barometer
			  ,DDH.RTU1_BatteryVoltage
			  ,DDH.Meter1_ID
			  ,DDH.Meter1_StaticPressPDayAvg
			  ,DDH.Meter1_Temperature
		FROM [EDW].[RTR].[DataDailyHistory] AS DDH
		WHERE DDH.Well1_Asset IN ('SJS')
			AND DDH.Well1_WellFlac = '""" + str(well_flac) +"""'
			AND DDH.Well1_WellFlac != 'FCFED2A'
			AND DDH.Well1_WellFlac IS NOT NULL;
	""")

	SQLCommand = ("""
		SELECT P.DateKey AS DateTime
			  ,W.WellFlac
			  ,W.Asset
			  ,W.WellName
		FROM [OperationsDataMart].[Facts].[Production] AS P
		JOIN [OperationsDataMart].[Dimensions].[Wells] AS W
			ON P.Wellkey = W.Wellkey
		WHERE P.DateKey >= '2017-01-01'
			AND W.WellFlac = '""" + str(well_flac) + """'
			AND W.Asset = 'Farmington';
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None

	df['WellName'] = df['WellName'].str.lower()
	df['DateTime'] = pd.to_datetime(df['DateTime'])

	# Query for the surface failures on record for a specific well
	failure_df = failures_fetch(well_flac)

	# fail_dic = {flac: failure_df[failure_df['WellFlac'] == flac]['fail_date'].values \
	# 			for flac in failure_df['WellFlac'].unique()}
    #
	# def fails_dates(row):
	# 	return (row['DateTime'] in fail_dic[row['WellFlac']])

	# Function to be applied to current day to calculate the last fail date
	def last_date(row):
		# print(row['WellFlac'])
		early_dates = failure_df[(failure_df['fail_date'] <= row['DateTime']) & \
								 (failure_df['WellFlac'] == row['WellFlac'])]['fail_date'].values
		if len(early_dates) > 0:
			return np.max(early_dates)
		else:
			return np.nan

	# Calculate last failure, the actual day it fails, and days since last fail
	# This is very time consuming when running across every well...
	# Is there a more efficient way to do this?
	# I can join the surface failures
	df['last_failure'] = df.apply(last_date, axis=1)
	df['last_failure'] = pd.to_datetime(df['last_failure'])

	df['failure'] = np.where(df['last_failure'] == df['DateTime'], 1, 0)
	df['days_since_fail'] = df['DateTime'] - df['last_failure']

	# Join compressor information
	df = compressor_link(df)

	# # Bring in the make and model of compressor for this well along with the
	# # total percentage of these compressors that fail
	# df['comp_model'] = fail_df[fail_df['WellFlac'] == well_flac]['make_model'].values[0]
	#
	# # Should this be calculated for the current specific date?
	# # Maybe pass model into a RF each time to use stacked model
	# df['percent_failure'] = fail_df[fail_df['WellFlac'] == well_flac]['fail_percentage'].values[0]
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
			  ,F.fail_count
		FROM   EDW.RTR.ConfigWell AS W
			INNER JOIN
			EDW.RTR.DataCurrentSnapshot AS S
			ON W.ID = S.ID
			LEFT OUTER JOIN
			OperationsDataMart.Dimensions.Wells AS DW
			ON W.WellFlac = DW.WellFlac
			LEFT OUTER JOIN
			#SurfaceFailures AS SF
			ON SF.assetWellFlac = W.WellFlac
			LEFT OUTER JOIN
				(SELECT COUNT(SF.surfaceFailureDate) AS fail_count
						 ,W.WellFlac
				 FROM #SurfaceFailures AS SF
				 JOIN EDW.RTR.ConfigWell AS W
				 ON SF.assetWellFlac = W.WellFlac
				 GROUP BY W.WellFlac) AS F
			ON F.WellFlac = W.WellFlac
		WHERE  W.WellStatus = 'A'
		   AND S.Comp1_Status != 'NA'
		   AND W.Asset IN ('SJS')
		   AND W.WellFlac = '"""+ str(well_flac) +"""'
		GROUP BY W.WellFlac, SF.surfaceFailureDate, W.Asset, W.ID, S.Comp1_Status, DW.WellName, F.fail_count;
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

def compressor_link(df):
	comps = pd.read_csv('data/compressors.csv', encoding = 'ISO-8859-1')
	comps['comp_model'] = comps['Compressor Manufacturer'].str.lower() + ' ' + comps['Compressor Model'].str.lower()
	comps['WellName'] = comps['Well Name'].str.lower()
	comps['WellName'] = comps['WellName'].str.replace('/', '_')
	comps_lim = comps[['WellName', 'Meter', 'comp_model']].dropna(how='all')

	# Merge surface failures with detailed compressor data
	joined = pd.merge(df, comps_lim, on='WellName', how='outer')

	# Create dummy for any failure
	# joined['fail'] = np.where(joined['fail_date'].notnull(), 1, 0)

	# Count total and percentage of failures for each make_model
	# fail_unique, fail_per = fail_count(joined)
	# joined['fail_unique'] = joined['make_model'].map(fail_unique)
	# joined['fail_percentage'] = joined['make_model'].map(fail_per)

	joined = joined[joined['WellFlac'].notnull()]
	# joined['WellFlac'] = joined['WellFlac'].astype(int)

	return joined

def make_model_pred(df, rf_model):
	make_model = df[df['comp_model'].notnull()]['comp_model'].values
	cols = np.loadtxt('data/unique_compressors.csv', delimiter=',', dtype='str')
	X = pd.DataFrame(np.full((len(make_model), len(cols)), 0), columns=cols)
	for idx, model in enumerate(make_model):
		X.set_value(idx, model, 1)
	pred = rf.predict_proba(X)
	return pred[:,1]

def time_series_model(df, rf_model):
	# Function used to determine dependent variable based on whether or not the
	# compressor will fail within a week of the current date
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
	# Decide if we want this as a classification or predicted probability of failing

	# comp_pred = make_model_pred(df, rf_model)
	# df['model_prediction'] = model_pred[0]

	# Stack the 2 models and use logistic regression?
	# Or should I just include the dummied model feature in a single model?
	# df['days_since_fail'].fillna(pd.Timedelta('0 days'), inplace=True)
	# df[df['days_since_fail'].notnull()]['days_since_fail'].astype(int, copy=False)
	df['days_since_fail'] = pd.to_numeric(df['days_since_fail'], errors='ignore')

	# Train/test split based on a 70/30 split
	test_date = df.iloc[int(df.shape[0] * .7),:]['DateTime']

	train = df[df['DateTime'] < test_date]
	test = df[df['DateTime'] >= test_date]

	# Remove codependent/non-numeric variables
	train = train.drop(['DateTime', 'Asset', 'WellFlac', 'WellName', 'comp_model', 'last_failure'], axis=1)
	test = test.drop(['DateTime', 'Asset', 'WellFlac', 'WellName', 'comp_model', 'last_failure'], axis=1)

	y_train = train.pop('fail_in_week')
	y_test = test.pop('fail_in_week')

	# Are there other classification models to try here?

	rf = RandomForestClassifier()
	rf.fit(train, y_train)
	accuracy = rf.score(test, y_test)
	print('Accuracy of last RF model:\n{}'.format(accuracy))
	return df


if __name__ == '__main__':
	data = pd.read_csv('data/failures_2017.csv')
	for flac in data[data['fail_count'] > 0]['WellFlac'].unique():
		print(flac)
		df = rtr_fetch(flac)
		# df.to_csv('data/temp_data.csv')
		# df = pd.read_csv('data/temp_data.csv')
		rf = joblib.load('random_forest_model.pkl')
		df = time_series_model(df, rf)

	# df = rtr_fetch(70317101)
	# df.to_csv('data/temp_data.csv')
	# df = pd.read_csv('data/temp_data.csv')
	# rf = joblib.load('random_forest_model.pkl')
	# df = time_series_model(df, rf)
