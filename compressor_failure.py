import pandas as pd
import numpy as np
import pyodbc
import sys
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def rtr_fetch(well_flac):
	try:
		connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-Test-L48.BP.Com;Database=EDW;trusted_connection=yes')
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
			  ,DDH.Well1_ConditionCode
			  ,DDH.Well1_CasingPress
			  ,DDH.Well1_CasingPressPDayAvg
			  ,DDH.Well1_TubingPress
			  ,DDH.Well1_TubingPressPDayAvg
			  ,DDH.Meter1_OffGasTarget
			  ,DDH.RTU1_Barometer
			  ,DDH.RTU1_BatteryVoltage
			  ,DDH.Meter1_DiffPress
			  ,DDH.Meter1_DiffPressPDayAvg
			  ,DDH.Meter1_VolumeCDay
			  ,DDH.Meter1_VolumePDay
			  ,DDH.Meter1_StaticPress
			  ,DDH.Meter1_StaticPressPDayAvg
			  ,DDH.Meter1_FlowRate
			  ,DDH.Meter1_Temperature
			  ,DDH.Meter1_FlowTimeCDay
			  ,DDH.Meter1_FlowTimePDay
		      ,DDH.Comp1_RunTimeCDay
		      ,DDH.Comp1_RunTimePDay
		      ,DDH.Comp1_Discharge
			  ,DDH.Comp1_RPM
		      ,DDH.Comp1_RecycleOutput
		      ,DDH.Comp1_Suction
			  ,DDH.Comp1_Status
		      ,DDH.Comp1_PanelCode
		      ,DDH.Comp1_PanelCodeText
		      ,DDH.Comp1_PanelStatusCode
		      ,DDH.Comp1_PanelStatusText
		FROM [EDW].[RTR].[DataDailyHistory] AS DDH
		WHERE DDH.Well1_Asset IN ('SJS')
			--AND DDH.Well1_WellFlac = '""" + str(well_flac) +"""'
			AND DDH.Well1_WellFlac != 'FCFED2A'
			AND DDH.Well1_WellFlac IS NOT NULL;
	""")

	# Is production data better? Seems to go back further, need to talk to someone about this
	# SQLCommand = ("""
	# 	SELECT P.DateKey AS DateTime
	# 		  ,W.WellFlac
	# 		  ,W.Asset
	# 		  ,W.WellName
	# 	FROM [OperationsDataMart].[Facts].[Production] AS P
	# 	JOIN [OperationsDataMart].[Dimensions].[Wells] AS W
	# 		ON P.Wellkey = W.Wellkey
	# 	WHERE P.DateKey >= '2017-01-01'
	# 		--AND W.WellFlac = '""" + str(well_flac) + """'
	# 		AND W.Asset = 'Farmington';
	# """)

	# Using hourly RTR data
	SQLCommand = ("""
		SELECT DHH.DateTime
		      ,DHH.Well1_Asset
		      ,DHH.Well1_WellName
		      ,DHH.Well1_WellFlac
		      ,DHH.RTU1_BatteryVoltage
		      ,DHH.RTU1_BatteryVoltagePDayAvg
		      ,DHH.RTU1_ProgramRevision
		      ,DHH.Well1_AmbientTemp
		      ,DHH.Well1_CasingPress
		      ,DHH.Well1_CasingPressPDayAvg
		      ,DHH.Well1_TubingPress
		      ,DHH.Well1_TubingPressPDayAvg
		      ,DHH.Well1_PowerStatus
		      ,DHH.Meter1_OffGasTarget
		      ,DHH.Meter1_ProjectedOffTarget
		      ,DHH.Meter1_VolumeCDay
		      ,DHH.Meter1_VolumePDay
		      ,DHH.Meter1_StaticPress
		      ,DHH.Meter1_StaticPressPDayAvg
		      ,DHH.Meter1_FlowRate
		      ,DHH.Meter1_Temperature
		      ,DHH.Meter1_TemperaturePDayAvg
		      ,DHH.Meter1_OffEngForecast
		      ,DHH.Meter1_Volume7DayAvg
		      ,DHH.Meter1_FlowTimeCDay
		      ,DHH.Meter1_FlowTimePDay
		      ,DHH.Comp1_RunTimeCDay
		      ,DHH.Comp1_RunTimePDay
		      ,DHH.Comp1_Discharge
		      ,DHH.Comp1_RPM
		      ,DHH.Comp1_Suction
		      ,DHH.Comp1_Status
	  FROM [EDW].[RTR].[DataHourlyHistory] AS DHH
	  WHERE Well1_Asset IN ('SJS')
	  	AND DHH.Well1_WellFlac IS NOT NULL;
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None

	df['WellName'] = df['Well1_WellName'].str.lower()
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
		early_dates = failure_df[(failure_df['fail_date'] <= row['DateTime']) & \
								 (failure_df['WellFlac'] == row['WellFlac'])]['fail_date'].values
		if len(early_dates) > 0:
			return np.max(early_dates)
		else:
			return np.nan

	# Calculate last failure, the actual day it fails, and days since last fail
	# This is very time consuming when running across every well...
	# Is there a more efficient way to do this?
	if failure_df.empty:
		# For wells with no failures, set values to -999
		df['last_failure'] = -999
		df['failure'] = -999
		df['days_since_fail'] = -999
	else:
		df['last_failure'] = df.apply(last_date, axis=1)
		df['last_failure'] = pd.to_datetime(df['last_failure'])

		df['failure'] = np.where(df['last_failure'] == df['DateTime'], 1, 0)
		df['days_since_fail'] = df['DateTime'] - df['last_failure']

	# Join compressor information
	df = compressor_link(df)

	df['days_since_fail'].fillna(-999, inplace=True)
	df.dropna(axis=0, subset=['comp_model'], inplace=True)

	df['DateTime'] = pd.to_datetime(df['DateTime'])

	def fail_in(row):
		days = 3
		if np.mean(df[(df['DateTime'] > row['DateTime']) & \
					  (df['DateTime'] <= (row['DateTime'] + \
					  datetime.timedelta(days=days))) & \
					  (df['WellFlac'] == row['WellFlac'])]['failure']) > 0:
			return 1
		else:
			return 0

	df['fail_in_week'] = df.apply(fail_in, axis=1)
	df['days_since_fail'] = pd.to_numeric(df['days_since_fail'], errors='ignore')

	df.columns = [col.lower().replace(' ', '_') for col in df.columns]

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
		   --AND W.WellFlac = '"""+ str(well_flac) +"""'
		GROUP BY W.WellFlac, SF.surfaceFailureDate, W.Asset, W.ID, S.Comp1_Status, DW.WellName, F.fail_count;
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

	return df

def compressor_link(df):
	'''
	Joins inputted dataframe with data on the compressors installed for each
	well within the dataframe.

	INPUT:
		df - Pandas dataframe containing WellFlacs

	OUTPUT:
		Joined DataFrame
	'''
	comps = pd.read_csv('data/compressors.csv', encoding = 'ISO-8859-1')
	comps['comp_model'] = comps['Compressor Manufacturer'].str.lower() + ' ' + comps['Compressor Model'].str.lower()
	comps['WellName'] = comps['Well Name'].str.lower()
	comps['WellName'] = comps['WellName'].str.replace('/', '_')
	comps_lim = comps[['WellName', 'Lwo Suction Kill', 'High suction kill', \
					   'Low Discharge Kill', 'High Discharge Kill', \
					   'Discharge temp kill', 'comp_model']].dropna(how='all')

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
	'''
	Utilized an inputted, trained random forest model to return the predicted
	probability that this model with fail.

	INPUTS:
		df - Pandas dataframe with data containing the make and model of the
			 compressor for each well.
		rf_model - Random forest classifier built from dummied variables of the
				   make and model of compressors.

	OUTPUTS:
		Numpy array containing the predicted probabilities that a certain
		compressor will fail.
	'''
	make_model = df[df['comp_model'].notnull()]['comp_model'].values
	cols = np.loadtxt('data/unique_compressors.csv', delimiter=',', dtype='str')
	X = pd.DataFrame(np.full((len(make_model), len(cols)), 0), columns=cols)
	for idx, model in enumerate(make_model):
		X.set_value(idx, model, 1)
	pred = rf.predict_proba(X)
	return pred[:,1]

def time_series_model(df, rf_model):
	'''
	Builds out predictor based on whether or not a compressor will fail within
	a designated number of days.
	Encorporates a pre-trained random forest classifier to take advantage of
	predicted probability of failure based solely on make and model of the
	compressor.
	Trains a random forest classifier for a final prediction on whether or not
	a compressor will fail over the next 7 days.

	INPUTS:
		df - Pandas dataframe containing daily information to be analyzed.
		rf_model - Random forest classifier built from dummied variables of the
				   make and model of compressors.

	OUTPUTS:
		Pandas dataframe containing the new predictor.
		Accuracy from the current run of the classifier.
	'''
	# Function used to determine dependent variable based on whether or not the
	# compressor will fail within a week of the current date
	df['datetime'] = pd.to_datetime(df['datetime'])

	def fail_in(row):
		days = 7
		if np.mean(df[(df['datetime'] > row['datetime']) & \
					  (df['datetime'] <= (row['datetime'] + \
					  datetime.timedelta(days=days))) & \
					  (df['wellflac'] == row['wellflac'])]['failure']) > 0:
			return 1
		else:
			return 0

	# df['fail_in_week'] = df.apply(fail_in, axis=1)

	# Get the predicted probability of failure based on compressor make/model
	# comp_pred = make_model_pred(df, rf_model)
	# df['model_prediction'] = comp_pred

	# Stack the 2 models and use logistic regression?
	# Or should I just include the dummied model feature in a single model?
	df['days_since_fail'] = pd.to_numeric(df['days_since_fail'], errors='ignore')

	# Train/test split based on a 70/30 split
	test_date = df.iloc[int(df.shape[0] * .7),:]['datetime']

	train = df[df['datetime'] < test_date].dropna(how='any', axis=1)
	test = df[df['datetime'] >= test_date].dropna(how='any', axis=1)

	# Remove codependent/non-numeric variables
	train = train.drop(['datetime', 'asset', 'failure', 'wellflac', 'wellname', \
						'comp_model', 'Unnamed: 0'], axis=1)
	test = test.drop(['datetime', 'asset', 'failure', 'wellflac', 'wellname', \
					  'comp_model', 'Unnamed: 0'], axis=1)
	# train = train[['days_since_fail', 'fail_in_week']]
	# test = test[['days_since_fail', 'fail_in_week']]

	y_train = train.pop('fail_in_week')
	y_test = test.pop('fail_in_week')

	sm = SMOTE(random_state=11)
	train, y_train = sm.fit_sample(train, y_train)

	# print(train.columns)
	# Are there other classification models to try here?
	rf = RandomForestClassifier(n_estimators=8, class_weight={0:1000, 1:1}, random_state=213)
	rf.fit(train, y_train)
	# print('Importances')
	# print(rf.feature_importances_)
	accuracy = rf.score(test, y_test)
	print('Accuracy of last RF model:\n{}'.format(accuracy))
	print('F1 Score:\n{}'.format(f1_score(y_test, rf.predict(test))))
	print('Precision Score:\n{}'.format(precision_score(y_test, rf.predict(test))))
	print('Recall Score:\n{}'.format(recall_score(y_test, rf.predict(test))))
	return df, accuracy

def logistic(df):
	# Function used to determine dependent variable based on whether or not the
	# compressor will fail within a week of the current date

	# Train/test split based on a 70/30 split
	test_date = df.iloc[int(df.shape[0] * .7),:]['datetime']

	train = df[df['datetime'] < test_date].dropna(how='any', axis=1)
	test = df[df['datetime'] >= test_date].dropna(how='any', axis=1)

	# Remove codependent/non-numeric variables
	train = train.drop(['datetime', 'asset', 'failure', 'wellflac', 'wellname', \
						'comp_model', 'Unnamed: 0'], axis=1)
	test = test.drop(['datetime', 'asset', 'failure', 'wellflac', 'wellname', \
					  'comp_model', 'Unnamed: 0'], axis=1)

	y_train = train.pop('fail_in_week')
	y_test = test.pop('fail_in_week')

	sm = SMOTE(random_state=11)
	x_train, y_train = sm.fit_sample(train, y_train)

	lr = LogisticRegression(class_weight={0:1, 1:1.333}, random_state=45)
	lr.fit(x_train, y_train)

	accuracy = lr.score(test, y_test)
	print('Accuracy of last RF model:\n{}'.format(accuracy))
	print('F1 Score:\n{}'.format(f1_score(y_test, lr.predict(test))))
	print('Precision Score:\n{}'.format(precision_score(y_test, lr.predict(test))))
	print('Recall Score:\n{}'.format(recall_score(y_test, lr.predict(test))))

	coeff_plot(list(train.columns), lr)

	return lr

def coeff_plot(feat, log_model):
	plt.close()

	fig, ax = plt.subplots(1, 1, figsize=(17, 15))
	# matplotlib.rcParams.update({'font.size': 18})

	ind = np.arange(len(feat) - 1)
	width = 0.35

	coeffs = list(log_model.coef_[0])
	del coeffs[feat.index('days_since_fail')]
	del feat[feat.index('days_since_fail')]

	p1 = ax.bar(ind, coeffs, width, color='#ba0025')
	ax.axhline()
	ax.set_ylabel('Coefficient Value')
	ax.set_xlabel('Feature')
	plt.xticks(ind, feat, rotation='vertical')

	plt.title('Feature Correlation Strength and Direction')
	plt.tight_layout()

	plt.savefig('images/lr_coef.png')

if __name__ == '__main__':
	# data = pd.read_csv('data/failures_2017.csv')
	# data[data['fail_count'] > 0]['WellFlac'].unique()
	# accs = []
	# for flac in [93229101]:
	# 	print(flac)
	# 	df = rtr_fetch(flac)
	# 	# df.to_csv('data/temp_data.csv')
	# 	# df = pd.read_csv('data/temp_data.csv')
	# 	rf = joblib.load('random_forest_model.pkl')
	# 	df, accuracy = time_series_model(df, rf)
	# 	accs.append(accuracy)
	# print('Average Accuracy: {}'.format(np.mean(accs)))

	df = rtr_fetch(70317101)
	df.to_csv('data/rtr_hourly_data_3.csv')
	# df = pd.read_csv('data/rtr_data_3.csv')
	# rf = joblib.load('random_forest_model.pkl')
	df, acc = time_series_model(df, rf)
	print('-------------------------------------------')
	lr_model = logistic(df)
