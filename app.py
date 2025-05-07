import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import io
import PIL.Image
import requests

# ── Fetch Stock Symbol ─────────────────────────────────────────────────────────
def get_stock_symbol(company_name):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json().get("quotes", [])
        if results:
            return results[0]['symbol'], results[0].get('shortname', company_name)
    return None, company_name

# ── Fetch Stock Data ──────────────────────────────────────────────────────────
def fetch_stock_data(ticker, period='10y'):
    df = yf.download(ticker, period=period, interval='1d')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    info = yf.Ticker(ticker).info
    metrics = {
        'Company': info.get('longName'),
        'Market Cap': info.get('marketCap'),
        'PE Ratio': info.get('trailingPE'),
        'EPS': info.get('trailingEps'),
        'Dividend Yield': info.get('dividendYield'),
        'ROE': info.get('returnOnEquity'),
        'Beta': info.get('beta')
    }
    return df, metrics

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("📊 AI Stock Investment Chatbot")
company_name = st.text_input("Enter Company Name", "Tesla")
period = st.selectbox("Select Time Period", ["1y", "2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y", "10y"], index=4)

if st.button("Predict Stock Price"):
    ticker, resolved_name = get_stock_symbol(company_name)
    
    if not ticker:
        st.error(f"❌ Couldn’t find ticker for '{company_name}'. Try again.")
    else:
        try:
            df, metrics = fetch_stock_data(ticker, period)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)
            
            def create_sequences(data, look_back=90):
                X = []
                for i in range(look_back, len(data)):
                    X.append(data[i - look_back:i])
                return np.array(X)

            if len(scaled_data) <= 90:
                st.warning("⚠️ Not enough data for LSTM prediction with selected time range.")
            else:
                X = create_sequences(scaled_data)
                X_latest = X[-1].reshape(1, 90, 5)

                model = load_model("lstm_model.h5")
                pred_scaled = model.predict(X_latest)
                pred_full = scaler.inverse_transform(
                    np.concatenate([np.zeros((1, 3)), pred_scaled, np.zeros((1, 1))], axis=1)
                )[0][3]

                # Plot the result
                plt.figure(figsize=(10, 5))
                df['Close'].plot(label=f"{period} Close Price")
                plt.axhline(pred_full, color='red', linestyle='--', label=f"Predicted Close: ${pred_full:.2f}")
                plt.title(f"{resolved_name} Closing Price Prediction ({period})")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = PIL.Image.open(buf)

                st.image(image)

                # Display Metrics
                st.markdown(f"**📈 Ticker:** {ticker}")
                st.markdown(f"**🏢 Company:** {metrics.get('Company', resolved_name)}")
                st.markdown(f"**💰 Predicted Closing Price (next day):** ${pred_full:.2f}")
                st.markdown(f"**📊 PE Ratio:** {metrics.get('PE Ratio')}")
                st.markdown(f"**🏦 Market Cap:** {metrics.get('Market Cap')}")
                st.markdown(f"**📈 EPS:** {metrics.get('EPS')}")
                st.markdown(f"**💸 Dividend Yield:** {metrics.get('Dividend Yield')}")
                st.markdown(f"**🚀 ROE:** {metrics.get('ROE')}")
                st.markdown(f"**⚖️ Beta:** {metrics.get('Beta')}")
                
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
