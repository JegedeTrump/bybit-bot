# bot.py
import os
import time
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('BybitBot')

# Load environment variables
load_dotenv()

class BybitTradingBot:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        
        # Configurable parameters
        self.leverage = 10
        self.risk_per_trade = 0.01  # 1% of portfolio per trade
        self.stop_loss = 0.015      # 1.5% stop loss
        self.take_profit = 0.03    # 3% take profit
        self.min_portfolio = 100    # Minimum portfolio value in USD
        self.scan_interval = 60     # Seconds between scans
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        self.symbols = self._get_available_symbols()
        
        # Strategy weights
        self.strategy_weights = {
            'bollinger': 0.35,
            'rsi': 0.25,
            'macd': 0.25,
            'volume': 0.15
        }
        
        logger.info("Bot initialized with %d trading pairs", len(self.symbols))

    def _initialize_exchange(self):
        return ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'rateLimit': 100  # ms between requests
        })

    def _get_available_symbols(self):
        markets = self.exchange.load_markets()
        return [
            symbol for symbol in markets 
            if '/USDT' in symbol and markets[symbol]['active']
        ]

    def get_portfolio_value(self):
        """Get total portfolio value in USDT"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['total'])
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return self.min_portfolio  # Fallback value

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        """Fetch OHLCV data with retry logic"""
        for _ in range(3):  # Retry up to 3 times
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df.set_index('timestamp')
            except Exception as e:
                logger.warning(f"Fetch failed for {symbol}: {e}, retrying...")
                time.sleep(2)
        return None

    # Pure Python technical indicators
    def calculate_bbands(self, close, period=20, std_dev=2):
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def calculate_rsi(self, close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_indicators(self, df):
        # Calculate all indicators
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bbands(df['close'])
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['signal'] = self.calculate_macd(df['close'])
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] / df['volume_ma']
        return df.dropna()

    def evaluate_strategies(self, df):
        """Evaluate trading strategies based on indicators"""
        signals = {}
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bollinger Bands strategy
        signals['bollinger'] = 0
        if last['close'] < last['bb_lower']:
            signals['bollinger'] = 1  # Oversold, buy signal
        elif last['close'] > last['bb_upper']:
            signals['bollinger'] = -1  # Overbought, sell signal
        
        # RSI strategy
        signals['rsi'] = 0
        if last['rsi'] < 30 and prev['rsi'] >= 30:
            signals['rsi'] = 1
        elif last['rsi'] > 70 and prev['rsi'] <= 70:
            signals['rsi'] = -1
        
        # MACD strategy
        signals['macd'] = 0
        if last['macd'] > last['signal'] and prev['macd'] <= prev['signal']:
            signals['macd'] = 1  # Bullish crossover
        elif last['macd'] < last['signal'] and prev['macd'] >= prev['signal']:
            signals['macd'] = -1  # Bearish crossover
        
        # Volume spike strategy
        signals['volume'] = 1 if last['volume_spike'] > 1.5 else 0
        
        return signals

    def calculate_score(self, signals):
        """Calculate weighted composite score"""
        return sum(
            self.strategy_weights[k] * v 
            for k, v in signals.items()
        )

    def find_trade_opportunity(self):
        """Scan all symbols for trading opportunities"""
        portfolio_value = max(self.get_portfolio_value(), self.min_portfolio)
        position_size = portfolio_value * self.risk_per_trade
        
        for symbol in self.symbols:
            try:
                df = self.fetch_ohlcv(symbol)
                if df is None or len(df) < 50:
                    continue
                    
                df = self.calculate_indicators(df)
                signals = self.evaluate_strategies(df)
                score = self.calculate_score(signals)
                
                if abs(score) >= 0.6:  # Minimum confidence threshold
                    direction = 1 if score > 0 else -1
                    logger.info(f"Opportunity found: {symbol} | Score: {score:.2f} | {'LONG' if direction == 1 else 'SHORT'}")
                    return symbol, direction, position_size
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        return None, None, None

    def execute_trade(self, symbol, direction, position_size):
        """Execute trade with proper risk management"""
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            if not price:
                logger.error(f"Failed to get price for {symbol}")
                return False
                
            # Calculate position size
            contract_size = float(self.exchange.market(symbol)['contractSize'])
            qty = (position_size * self.leverage) / (price * contract_size)
            qty = round(qty, self.exchange.market(symbol)['precision']['amount'])
            
            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol)
            
            # Calculate stop loss and take profit
            sl_price = price * (1 - self.stop_loss) if direction == 1 else price * (1 + self.stop_loss)
            tp_price = price * (1 + self.take_profit) if direction == 1 else price * (1 - self.take_profit)
            
            # Place order
            side = 'buy' if direction == 1 else 'sell'
            order = self.exchange.create_order(
                symbol=symbol,
                type='MARKET',
                side=side,
                amount=qty,
                params={
                    'stopLoss': sl_price,
                    'takeProfit': tp_price
                }
            )
            
            logger.info(f"Trade executed: {symbol} | {side} | Qty: {qty} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def run(self):
        logger.info("Starting trading bot...")
        while True:
            try:
                symbol, direction, size = self.find_trade_opportunity()
                
                if symbol:
                    self.execute_trade(symbol, direction, size)
                else:
                    logger.info(f"No opportunities found. Next scan in {self.scan_interval} seconds")
                
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Critical error in main loop: {e}")
                logger.info(f"Restarting in {self.scan_interval * 2} seconds")
                time.sleep(self.scan_interval * 2)

if __name__ == "__main__":
    bot = BybitTradingBot()
    bot.run()
