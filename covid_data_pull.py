import pandas as pd

def read_jhu(date):
    """Provide date in MM-DD-YYYY format"""
    
    url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/{date}.csv'

    data = pd.read_csv(url,usecols=['Province_State', 'Last_Update', 'Incident_Rate', 'Mortality_Rate', 'Testing_Rate', 'Hospitalization_Rate'], parse_dates=['Last_Update'])
    data = data.loc[~(d.Province_State == 'Diamond Princess')]
    data.Last_Update = data.Last_Update.dt.date
    
    return data

def combine_jhu(range_start, range_end):
    import requests
    start_url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/{range_start}.csv'
    end_url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/{range_end}.csv'

    start_response = requests.get(url=start_url).status_code
    end_response = requests.get(url=end_url).status_code
    
    if start_response != 200:
        print(f'Start Date Response {start_response}')
    
    if end_response != 200:
        print(f'End Date Response {end_response}')

    dates_list = [date.strftime('%m-%d-%Y') for date in pd.date_range(start=range_start, end=range_end)]

    frames = [read_jhu(date) for date in dates_list]
    result = pd.concat(frames)
    
    return result