from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging configuration for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask application with CORS support
app = Flask(__name__)
CORS(app)

class StockAnalyzer:
    """
    Handles all stock analysis operations including technical indicators,
    risk metrics, and trading signals.
    """
    
    @staticmethod
    def get_stock_data(ticker):
        """
        Fetches comprehensive stock data using yfinance API
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty:
                raise ValueError(f"No historical data found for {ticker}")
            
            info = stock.info
            
            # Get ESG data if available
            try:
                sustainability = stock.sustainability
                if sustainability is not None:
                    esg_data = sustainability.to_dict()
                else:
                    esg_data = None
            except:
                esg_data = None
            
            return hist, info, esg_data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

    @staticmethod
    def calculate_technical_indicators(hist_data):
        """
        Calculates various technical indicators from historical price data
        """
        df = hist_data.copy()
        
        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate Volatility (20-day standard deviation)
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        return df

    @staticmethod
    def calculate_risk_metrics(hist_data, current_price):
        """
        Calculates key risk metrics including VaR and position sizing
        """
        returns = hist_data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * current_price
        
        # Calculate suggested position size based on volatility
        base_position = 100000  # Base portfolio value
        position_size = base_position * (1 - volatility)
        
        # Calculate stop loss level (2 standard deviations)
        stop_loss = current_price * (1 - 2 * volatility)
        
        return {
            'volatility': volatility,
            'var_95': abs(var_95),
            'suggested_position_size': position_size,
            'stop_loss': stop_loss
        }

    @staticmethod
    def calculate_trading_signal(tech_data, esg_score=None):
        """
        Generates trading signals based on technical and ESG indicators
        """
        latest = tech_data.iloc[-1]
        
        # Technical signals
        trend_signal = 1 if latest['SMA_20'] > latest['SMA_50'] else -1
        rsi_signal = -1 if latest['RSI'] > 70 else 1 if latest['RSI'] < 30 else 0
        
        # Combine signals (add ESG component if available)
        if esg_score:
            esg_component = 1 if esg_score > 50 else -1
            final_signal = (trend_signal + rsi_signal + esg_component) / 3
        else:
            final_signal = (trend_signal + rsi_signal) / 2
        
        # Convert to simple signal (-1, 0, 1)
        signal = 1 if final_signal > 0.3 else -1 if final_signal < -0.3 else 0
        strength = min(abs(final_signal), 1)
        
        return signal, strength

    @staticmethod
    def simulate_backtest(hist_data):
        """
        Performs historical backtesting of the trading strategy
        """
        portfolio_values = []
        initial_value = 100000
        current_position = 0
        cash = initial_value
        trades = []
        
        # Calculate technical indicators for the entire period
        tech_data = StockAnalyzer.calculate_technical_indicators(hist_data)
        
        for i in range(50, len(hist_data)):  # Start after warmup period
            date = hist_data.index[i]
            price = hist_data['Close'].iloc[i]
            
            # Generate signal based on previous day's data
            signal, _ = StockAnalyzer.calculate_trading_signal(tech_data.iloc[:i])
            
            # Execute trades based on signals
            if signal == 1 and current_position == 0:  # Buy signal
                shares = cash / price
                cash = 0
                current_position = shares
                trades.append({
                    'date': date.isoformat(),
                    'type': 'buy',
                    'price': price,
                    'shares': shares
                })
            elif signal == -1 and current_position > 0:  # Sell signal
                cash = current_position * price
                trades.append({
                    'date': date.isoformat(),
                    'type': 'sell',
                    'price': price,
                    'shares': current_position,
                    'pnl': cash - initial_value
                })
                current_position = 0
            
            # Track portfolio value
            portfolio_values.append({
                'date': date.isoformat(),
                'value': cash + (current_position * price if current_position > 0 else 0)
            })
        
        # Calculate final metrics
        final_value = portfolio_values[-1]['value']
        returns = (final_value - initial_value) / initial_value
        volatility = np.std([pv['value'] for pv in portfolio_values]) / initial_value
        sharpe_ratio = (returns - 0.02) / volatility if volatility != 0 else 0
        
        return {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'final_capital': final_value,
            'returns': returns,
            'sharpe_ratio': sharpe_ratio
        }

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main endpoint for stock analysis
    """
    try:
        # Get and validate ticker from request
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({"error": "No ticker provided"}), 400
            
        ticker = data['ticker'].upper()
        logger.info(f"Analyzing ticker: {ticker}")
        
        # Get stock data
        hist_data, stock_info, esg_data = StockAnalyzer.get_stock_data(ticker)
        current_price = hist_data['Close'].iloc[-1]
        
        # Calculate technical indicators
        tech_data = StockAnalyzer.calculate_technical_indicators(hist_data)
        latest_tech = tech_data.iloc[-1]
        
        # Calculate risk metrics
        risk_metrics = StockAnalyzer.calculate_risk_metrics(hist_data, current_price)
        
        # Generate trading signal
        signal, strength = StockAnalyzer.calculate_trading_signal(tech_data)
        
        # Run backtest
        backtest_results = StockAnalyzer.simulate_backtest(hist_data)
        
        # Prepare response
        response_data = {
            'ticker': ticker,
            'esg_components': esg_data,
            'financial_metrics': {
                'pe_ratio': stock_info.get('forwardPE', 0),
                'market_cap': stock_info.get('marketCap', 0),
                'dividend_yield': stock_info.get('dividendYield', 0) or 0,
                'debt_to_equity': stock_info.get('debtToEquity', 0),
                'current_price': current_price
            },
            'technical_indicators': {
                'RSI': latest_tech['RSI'],
                'SMA_20': latest_tech['SMA_20'],
                'SMA_50': latest_tech['SMA_50'],
                'Volatility': latest_tech['Volatility']
            },
            'risk_metrics': risk_metrics,
            'trading_signal': signal,
            'signal_strength': strength,
            'backtest_results': backtest_results
        }
        
        logger.info("Analysis completed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)