import streamlit as st
from datetime import date, datetime

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go # a library to get interactive graphs

# retrieve data from dates
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "GOOGL", "INTU", "TSLA")
selected_stock = st.selectbox("Select data for prediction", stocks)

num_years = st.slider("Years of prediction:", 1, 5) # widget slider, a label, and start and end value
period = num_years * 365

@st.cache # the following function will be cached
def load_data(ticker):
    data = yf.download(ticker, START, TODAY) # parameters
    data.reset_index(inplace=True) # puts date in first column
    return data

data_load_state = st.text("Load data...") # makes the webpage interactive
data = load_data(selected_stock)
data_load_state.text("Loading data...done") # this widget resets widget above

st.subheader("Raw Data")
st.caption("Displays the last five market days")
st.write(data.tail()) # grabs the tail of the data frame

# plotting data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail()) # the last ten rows

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

