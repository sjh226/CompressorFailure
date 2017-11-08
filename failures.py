import pandas as pd
import numpy as np
import pyodbc
import sys
import re
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_flacs():
	try:
		connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-L48.BP.Com;Database=OperationsDataMart;trusted_connection=yes')
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

def comp_link(limit=False):
	# Data on compressor-specific surface failures
	failures = pd.read_csv('surface_failures.csv').drop_duplicates()
	failures['well_name'] = failures['Well1_WellName'].str.lower()
	failures['well_name'] = failures['well_name'].str.replace('/', '_')
	failures['fail_date'] = pd.to_datetime(failures['surfaceFailureDate'])
	fail_lim = failures[['WellFlac', 'fail_date', 'well_name']]

	if limit == True:
		# Limit to one month's worth of data
		fail_lim = fail_lim[((fail_lim['fail_date'] >= np.min(fail_lim['fail_date'])) & \
					   		(fail_lim['fail_date'] >= np.min(fail_lim['fail_date']) + datetime.timedelta(days=30))) | \
					   		(fail_lim['fail_date'].isnull())]

	# Data on all compressors
	comps = pd.read_csv('compressors.csv', encoding = 'ISO-8859-1')
	comps['make_model'] = comps['Compressor Manufacturer'].str.lower() + ' ' + comps['Compressor Model'].str.lower()
	comps['well_name'] = comps['Well Name'].str.lower()
	comps['well_name'] = comps['well_name'].str.replace('/', '_')
	comps_lim = comps[['well_name', 'Meter', 'make_model']].dropna(how='all')

	# Bring in information on all wells
	try:
		wells = pd.read_csv('well_relations.csv')
	except:
		wells = get_flacs()
		wells['well_name'] = wells['WellName'].str.lower()
		wells['well_name'] = wells['well_name'].str.replace('/', '_')
		wells.to_csv('well_relations.csv')

	# Merge compressor data with well info
	# 95 wells do not have compressor information
	# 64 unique compressors
	comps_lim_full = pd.merge(comps_lim, wells, on='well_name')

	# Merge surface failures with detailed compressor data
	joined = pd.merge(fail_lim, comps_lim_full, on='well_name', how='outer')
	joined['WellFlac'] = np.where(joined['WellFlac_x'].notnull(), joined['WellFlac_x'],\
								  joined['WellFlac_y']).astype(int)
	joined = joined[['WellFlac', 'well_name', 'Meter', 'make_model', 'fail_date']]

	# Create dummy for any failure
	joined['fail'] = np.where(joined['fail_date'].notnull(), 1, 0)
	joined.drop_duplicates(inplace=True)

	# Create variable for whether it fails in the last week
	joined['fail_pred'] = np.where(joined['fail_date'] >= np.max(joined['fail_date'] - \
								   datetime.timedelta(days = 7)), 1, 0)

	# Count total and percentage of failures for each make_model
	fail_unique, fail_per, fail_totals = fail_count(joined)
	joined['fail_unique'] = joined['make_model'].map(fail_unique)
	joined['fail_percentage'] = joined['make_model'].map(fail_per)
	joined['fail_totals'] = joined['make_model'].map(fail_totals)
	return joined

def fail_count(df):
	fail_unique = {}
	fail_per = {}
	fail_totals = {}
	for model in df['make_model'].unique():
		count = len(df[(df['make_model'] == model) & (df['fail'] == 1)]['WellFlac'].unique())
		total = len(df[df['make_model'] == model]['WellFlac'].unique())
		fail_unique[model] = count
		fail_totals[model] = df[(df['make_model'] == model) & (df['fail'] == 1)].shape[0]
		try:
			fail_per[model] = count/total
		except ZeroDivisionError:
			fail_per[model] = 0
	return fail_unique, fail_per, fail_totals

# Need to compare failures will compressors that do not fail
# Look at percentage of each make/model that fail
# Use compressors that don't fail somehow?
# Could look into what the scheduled maintenance looks like for those that don't fail
# Can we see frequency of failure?

def failure_classifier(df, split='normal'):
	if split == 'time':
		# This is tricky since there is no dependence on time
		train = df[df['fail_date'] >= np.max(df['fail_date']) - datetime.timedelta(days=7)]
		test = df[df['fail_date'] < np.max(df['fail_date']) - datetime.timedelta(days=7)]

		X_train = pd.get_dummies(train['make_model'])
		X_test = pd.get_dummies(train['make_model'])

		y_train = train['fail_pred']
		y_test = train['fail_pred']
	else:
		X = pd.get_dummies(df['make_model'])
		y = df['fail']
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=87)

	# Should grid search for best hyperparameters
	rf = RandomForestClassifier()
	rf.fit(X_train, y_train)
	accuracy = rf.score(X_test, y_test)

	model_probs = pd.get_dummies(df['make_model'].unique())
	pred_prob = rf.predict_proba(model_probs)
	probs = {}
	for i, model in enumerate(df['make_model'].unique()):
		probs[model] = pred_prob[i, 1]

	print('Accuracy of last run: {:.2f}\n'.format(accuracy))
	# Out of box ~75%

	comp_importance = dict(zip(X, rf.feature_importances_))
	# Find the worst compressor
	worst = max(comp_importance.keys(), key=lambda key: comp_importance[key])
	print('Worst Compressor: {}'.format(worst))
	print('{} total failures.'.format(df[df['make_model'] == worst]['fail_totals'].unique()[0]))
	print('Failed on {} unique wells.'.format(df[df['make_model'] == worst]['fail_unique'].unique()[0]))
	print('Installed on {} wells.'.format(len(df[df['make_model'] == worst]['WellFlac'].unique())))
	print('{:.2f}% of these compressors fail.'.format(len(df[(df['make_model'] == worst) & \
													 (df['fail'] == 1)]['WellFlac'].unique()) / \
													 len(df[df['make_model'] == worst]['WellFlac'].unique())))

	# Worst Compressor: LeRoi HFG12000
	# 59 total failures.
	# Failed on 37 unique wells.
	# Installed on 125 wells.
	# 30% of these compressors fail.

	return rf, comp_importance, probs

def fail_stats(df, probs):
	# Drop NANs (wells without compressors)
	df.dropna(subset=['make_model'], inplace=True)

	model_stats = pd.DataFrame(columns=['make_model', 'total_failures', \
										'fail_percentage', 'unique_failures', \
										'well_count', 'pred_prob_fail'])
	for model in df['make_model'].unique():
		well_count = len(df[df['make_model'] == model]['WellFlac'].unique())
		tot_fail = df[df['make_model'] == model]['fail_totals'].unique()[0]
		perc_fail = df[df['make_model'] == model]['fail_percentage'].unique()[0]
		uniq_fail = df[df['make_model'] == model]['fail_unique'].unique()[0]
		prob = probs[model]
		stats = pd.DataFrame([[model, tot_fail, perc_fail, uniq_fail, well_count, prob]], \
							 columns=model_stats.columns)

		model_stats = model_stats.append(stats)

	model_stats.sort_values('pred_prob_fail', ascending=False, inplace=True)
	model_stats.to_csv('stats.csv')
	return model_stats


if __name__ == '__main__':
	# This is currently limited to early September, do we have data before then?
	fail_df = comp_link()
	fail_rf, imp, pred_prob = failure_classifier(fail_df)
	stats = fail_stats(fail_df, pred_prob)
