import pandas as pd
from fredapi import Fred

print("pandas:", pd.__version__)
try:
    import fredapi
    print("fredapi imported")
except Exception as e:
    print("fredapi import error:", e)

fred = Fred(api_key='e1ab1d32d6233f0589e41d8a74f37174')
# Collect data
unemployment = fred.get_series('UNRATE')
cpi = fred.get_series('CPIAUCSL')
gdp = fred.get_series('GDP')
# Combine into dataframe
df = pd.DataFrame({
	'unemployment': unemployment,
	'cpi': cpi,
	'gdp': gdp
})