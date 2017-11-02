import pandas as pd
import numpy as np


def comp_link():
    failures = pd.read_csv('surface_failures.csv').drop_duplicates()
    comps = pd.read_csv('compressors.csv', encoding = 'ISO-8859-1')
    # Pull WellFlacs into comps

    fail_lim = failures[['WellFlac', 'surfaceFailureDate', 'Well1_WellName', \
                         ]].set_index('Well1_WellName')
    comps['make_model'] = comps['Compressor Manufacturer'] + ' ' + comps['Compressor Model']
    comps_lim = comps[['Well Name', 'Meter', 'make_model']].set_index('Well Name')

    joined = fail_lim.join(comps_lim, how='outer')
    return joined.drop_duplicates()

# Need to compare failures will compressors that do not fail
# Look at percentage of each make/model that fail
# Use compressors that don't fail somehow?
# Could look into what the scheduled maintenance looks like for those that don't fail
# Can we see frequency of failure?


if __name__ == '__main__':
    df = comp_link()
