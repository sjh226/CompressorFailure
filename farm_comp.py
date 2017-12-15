import numpy as np
import pandas as pd
import pyodbc


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
			  ,CASE WHEN CH.RunStatus = 'Down' THEN 1
					ELSE 0
					END
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

	except:
		pass
        
	connection.close()

	return df


if __name__ == '__main__':
	df = data_fetch()
