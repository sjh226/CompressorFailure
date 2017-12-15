import numpy as np
import pandas as pd
import pyodbc
import datetime


def data_fetch():
	try:
		connection = pyodbc.connect(r'DRIVER={SQL Server Native Client 11.0};'
									r'SERVER=149.181.83.106;'
									r'DATABASE=Runtime;'
									r'UID=adhoc;'
									r'PWD=adhoc'
									)
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()

	# Create temporary table for compressor failures
	# Currently focused on Farmington
	SQLCommand = ("""
		SELECT CH.ID
			  ,CH.Meter
			  ,CH.Name
			  ,CH.RecordDate
			  ,CASE WHEN CH.RunStatus = 'Down' THEN 1
					ELSE 0
					END AS DownStatus
			  ,CH.CompressorType
			  ,CH.ConditionCode
			  ,CH.GasVolumePDay
			  ,CH.GasFlowRate
			  ,CH.RPM
			  ,CH.RunTimeCDay
			  ,CH.RunTimePDay
			  ,CH.SuctionPressure
			  ,CH.DischargePressure
			  ,CH.InterstagePressure
			  ,CH.LinePressure
			  ,CH.FuelGasPressure
			  ,CH.ManifoldPressure
			  ,CH.EngineOilPressure
			  ,CH.FrameOilPressure
			  ,CH.PreCoolerDischargeTemp
			  ,CH.AfterCoolerDischargeTemp
			  ,CH.FinalDischargeTemp
			  ,CH.EngineTempIn
			  ,CH.EngineTempOut
			  ,CH.FrameOilTemp
			  ,CH.EngineVibration
			  ,CH.FrameVibration
			  ,CH.SuctionPosition
			  ,CH.RecyclePosition
			  ,CH.RateshedOutput
			  ,CH.AutoStartsCDay
			  ,CH.AutoStartsPDay
			  ,CH.AutoAttemptsCDay
			  ,CH.AutoAttemptsPDay
		FROM   Runtime.dbo.CompressorHistory AS CH;
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
		df.columns = [col.lower().replace(' ', '_') for col in df.columns]
	except:
		pass

	connection.close()

	return df

def analyze(df):
	for meter in df['meter'].unique():
		unq = df[df['meter'] == meter]['recorddate'].value_counts()
		print(meter, ' ', unq[-1])

def model(df):
	def fail_in(row):
		days = 3
		if np.mean(df[(df['recorddate'] > row['recorddate']) & \
					  (df['recorddate'] <= (row['recorddate'] + \
					  datetime.timedelta(days=days))) & \
					  (df['meter'] == row['meter'])]['downstatus']) > 0:
			return 1
		else:
			return 0

	df['fail_in_week'] = df.apply(fail_in, axis=1)

	return df


if __name__ == '__main__':
	df = data_fetch()
	# analyze(df)
	df = model(df)
