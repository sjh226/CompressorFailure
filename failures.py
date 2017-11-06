import pandas as pd
import numpy as np
import pyodbc
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_flacs():
	try:
		connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=OperationsDataMart;trusted_connection=yes')
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		SELECT W.Wellkey
			  ,W.WellName
			  ,W.WellFlac
		FROM [OperationsDataMart].[Dimensions].[Wells] AS W;
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None

	return df

def comp_link():
	# Data on compressor-specific surface failures
	failures = pd.read_csv('surface_failures.csv').drop_duplicates()
	failures['WellName'] = failures['Well1_WellName'].str.replace('/','_')
	fail_lim = failures[['WellFlac', 'surfaceFailureDate', 'WellName']]

	# Data on all compressors
	comps = pd.read_csv('compressors.csv', encoding = 'ISO-8859-1')
	comps['make_model'] = comps['Compressor Manufacturer'] + ' ' + comps['Compressor Model']
	comps['WellName'] = comps['Well Name'].str.replace('/','_')
	comps_lim = comps[['WellName', 'Meter', 'make_model']].dropna(how='all')

	# Bring in information on all wells
	wells = get_flacs()
	wells['WellName'] = wells['WellName'].str.replace('/','_')

	# Merge compressor data with well info
	# 95 wells do not have compressor information
	comps_lim_full = pd.merge(comps_lim, wells, on='WellName')

	# Merge surface failures with detailed compressor data
	joined = pd.merge(fail_lim, comps_lim_full, on='WellName', how='outer')
	joined['WellFlac'] = np.where(joined['WellFlac_x'].notnull(), joined['WellFlac_x'],\
								  joined['WellFlac_y']).astype(int)
	joined = joined[['WellFlac', 'WellName', 'Meter', 'make_model', 'surfaceFailureDate']]

	# Create dummy for any failure
	joined['fail'] = np.where(joined['surfaceFailureDate'].notnull(), 1, 0)
	joined.drop_duplicates(inplace=True)

	# Count total and percentage of failures for each make_model
	fail_dic, fail_per = fail_count(joined)
	joined['fail_count'] = joined['make_model'].map(fail_dic)
	joined['fail_percentage'] = joined['make_model'].map(fail_per)
	return joined

def fail_count(df):
	fail_dic = {}
	fail_per = {}
	for model in df['make_model'].unique():
		count = len(df[(df['make_model'] == model) & (df['fail'] == 1)]['WellFlac'].unique())
		total = len(df[df['make_model'] == model]['WellFlac'].unique())
		fail_dic[model] = count
		try:
			fail_per[model] = count/total
		except ZeroDivisionError:
			fail_per[model] = 0
	return fail_dic, fail_per

# Need to compare failures will compressors that do not fail
# Look at percentage of each make/model that fail
# Use compressors that don't fail somehow?
# Could look into what the scheduled maintenance looks like for those that don't fail
# Can we see frequency of failure?

def failure_classifier(df):
	X = pd.get_dummies(df['make_model'])
	y = df['fail']

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=87)

	# Should grid search for best hyperparameters
	rf = RandomForestClassifier()
	rf.fit(X_train, y_train)
	accuracy = rf.score(X_test, y_test)

	print('Accuracy of last run: {:.2f}\n'.format(accuracy))
	# Out of box ~75%

	comp_importance = dict(zip(X, rf.feature_importances_))
	# Find the worst compressor
	worst = max(comp_importance.keys(), key=lambda key: comp_importance[key])
	print('Worst Compressor: {}'.format(worst))
	print('{} failures.'.format(df[df['make_model'] == worst]['fail_count'].unique()[0]))
	print('Installed on {} wells.'.format(len(df[df['make_model'] == worst]['WellFlac'].unique())))
	print('{:.2f}% of these compressors fail.'.format(len(df[(df['make_model'] == worst) & \
													 (df['fail'] == 1)]['WellFlac'].unique()) / \
													 len(df[df['make_model'] == worst]['WellFlac'].unique())))

	# Worst Compressor: LeRoi HFG12000
	# 57 failures.
	# Installed on 116 wells.
	# 30% of these compressors fail.

	return rf, comp_importance


if __name__ == '__main__':
	fail_df = comp_link()
	fail_rf, imp = failure_classifier(fail_df)
