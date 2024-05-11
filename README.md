
### Streamlit

https://fiap-oil-price-prediction.streamlit.app/

### Main Files to analyze

- `data/brent_crude_oil_prices.csv` - This is the main dataset that was used for the analysis.
  - To collect it I used the following code:
  ```python
    import pandas as pd

    df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', encoding='iso-8859-1', thousands='.', decimal=',')[2]
    # Use slicing to skip the first row
    df = df.iloc[1:].reset_index(drop=True)
    # Rename columns
    df.columns = ['Date', 'Price']
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    # Convert 'Price' to float or int as necessary
    df['Price'] = df['Price'].astype(float)
    # Sort by 'Date' in descending order
    df = df.sort_values('Date', ascending=False)
    # save to csv
    df.to_csv('../data/brent_crude_oil_prices.csv', index=False)
  ```
- `notebook/analysis.ipynb` - This is where the main data analysis was done, and the LSTM model was created.
- `app.py` - This is the Streamlit app that was created to visualize the data, and show the model being used.