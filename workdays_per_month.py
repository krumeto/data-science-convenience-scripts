import pandas as pd
from workalendar.usa import UnitedStates
from datetime import date

def get_usa_working_days_per_month(start='2018-01-01', end='2022-01-01'):
    date_range = pd.period_range(start=start, end=end, freq='M')
    df = pd.DataFrame(date_range, columns=['year_month'])
    
    cal = UnitedStates()
    
    def wdays_per_month(month):
        return cal.get_working_days_delta(
            date(month.year, month.month, 1), 
            date(month.year, month.month, month.day), 
            include_start=True)
    
    df['working_days'] = df['year_month'].apply(wdays_per_month)

    return df


print(get_usa_working_days_per_month())
