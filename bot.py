import requests
from urllib.parse import quote
import os
import time
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
import sys
from typing import Dict, Optional, Tuple

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
        self.trade_count = 0
        self.total_profit = 0

        # Configurable parameters
        self.leverage = 10
        self.risk_per_trade = 0.01
        self.stop_loss = 0.015
        self.take_profit = 0.03
        self.min_portfolio = 100
        self.scan_interval = 60
        self.min_order_size = 5
        self.timeframe = '1h'
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'accountType': 'UNIFIED'
            }
        })
        
        # Test connection
        self._test_connection()
        self.symbols = self._get_available_symbols()
        
        # Enhanced Strategy Configuration
        self.strategy_weights = {
            'bollinger_bands_reversal': 0.25,
            'rsi_divergence': 0.20,
            'macd_crossover': 0.20,
            'volume_spike': 0.15,
            'ema_crossover': 0.10,
            'obv_breakout': 0.10
        }
        
        logger.info(f"Bot initialized with {len(self.symbols)} trading pairs")

    def send_telegram_alert(self, message: str):
        """Send message to Telegram channel"""
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                logger.warning("Telegram credentials not configured")
                return False
                
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            response = requests.post(url, json={'chat_id': chat_id, 'text': message})
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {str(e)}")
            return False

    def _test_connection(self):
        """Test exchange connection with unified account params"""
        try:
            balance = self.exchange.private_get_v5_account_wallet_balance({
                'accountType': 'UNIFIED'
            })
            usdt_balance = float(balance['result']['list'][0]['coin'][0]['walletBalance'])
            logger.info(f"Successfully connected to Bybit. USDT Balance: {usdt_balance:.2f}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            raise ConnectionError("Failed to connect to Bybit API")

    def _get_available_symbols(self):
        """Get available USDT perpetual contracts"""
        markets = self.exchange.load_markets()
        return [
            symbol for symbol in markets 
            if symbol.endswith('USDT') and 'swap' in markets[symbol]['type']
        ]

    def get_portfolio_value(self):
        """Get total portfolio value in USDT"""
        try:
            balance = self.exchange.private_get_v5_account_wallet_balance({
                'accountType': 'UNIFIED'
            })
            return float(balance['result']['list'][0]['coin'][0]['walletBalance'])
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return self.min_portfolio

    def fetch_ohlcv(self, symbol, timeframe=None, limit=100):
        """Fetch OHLCV data with retry logic"""
        timeframe = timeframe or self.timeframe
        for _ in range(3):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df.set_index('timestamp')
            except Exception as e:
                logger.warning(f"Fetch failed for {symbol}: {e}, retrying...")
                time.sleep(2)
        return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bbands(df['close'])
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['signal'], _ = self._calculate_macd(df['close'])
        
        # Volume Analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] / df['volume_ma']
        
        # EMA Cross
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # OBV
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        
        return df.dropna()

    def _calculate_bbands(self, close: pd.Series, period: int = 20, std_dev: int = 2):
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def _calculate_rsi(self, close: pd.Series, period: int = 14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_obv(self, close: pd.Series, volume: pd.Series):
        obv = [0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)

    def evaluate_strategies(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate all trading strategies"""
        signals = {}
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bollinger Bands Reversal
        signals['bollinger_bands_reversal'] = 0
        if last['close'] < last['bb_lower']:
            signals['bollinger_bands_reversal'] = 1
        elif last['close'] > last['bb_upper']:
            signals['bollinger_bands_reversal'] = -1
        
        # RSI Divergence
        signals['rsi_divergence'] = 0
        if last['rsi'] < 30 and prev['rsi'] >= 30:
            signals['rsi_divergence'] = 1
        elif last['rsi'] > 70 and prev['rsi'] <= 70:
            signals['rsi_divergence'] = -1
        
        # MACD Crossover
        signals['macd_crossover'] = 0
        if last['macd'] > last['signal'] and prev['macd'] <= prev['signal']:
            signals['macd_crossover'] = 1
        elif last['macd'] < last['signal'] and prev['macd'] >= prev['signal']:
            signals['macd_crossover'] = -1
        
        # Volume Spike
        signals['volume_spike'] = 1 if last['volume_spike'] > 1.5 else 0
        
        # EMA Crossover
        signals['ema_crossover'] = 0
        if last['ema_short'] > last['ema_long'] and prev['ema_short'] <= prev['ema_long']:
            signals['ema_crossover'] = 1
        elif last['ema_short'] < last['ema_long'] and prev['ema_short'] >= prev['ema_long']:
            signals['ema_crossover'] = -1
        
        # OBV Breakout
        signals['obv_breakout'] = 0
        obv_ma = df['obv'].rolling(20).mean()
        if last['obv'] > obv_ma[-1] and prev['obv'] <= obv_ma[-2]:
            signals['obv_breakout'] = 1
        elif last['obv'] < obv_ma[-1] and prev['obv'] >= obv_ma[-2]:
            signals['obv_breakout'] = -1
        
        return signals

    def calculate_composite_score(self, signals: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        return sum(
            self.strategy_weights[strat] * signal 
            for strat, signal in signals.items()
        )

    def find_trade_opportunity(self) -> Optional[Tuple[str, int, float]]:
        """Scan all symbols for trading opportunities and send preview alerts"""
        portfolio_value = max(self.get_portfolio_value(), self.min_portfolio)
        position_size = portfolio_value * self.risk_per_trade
        
        if position_size < self.min_order_size:
            logger.warning(f"Position size too small: {position_size:.2f} USDT")
            return None, None, None
            
        for symbol in self.symbols:
            try:
                df = self.fetch_ohlcv(symbol)
                if df is None or len(df) < 50:
                    continue
                    
                df = self.calculate_indicators(df)
                signals = self.evaluate_strategies(df)
                score = self.calculate_composite_score(signals)
                
                if abs(score) >= 0.6:
                    direction = 1 if score > 0 else -1
                    side = 'LONG' if direction == 1 else 'SHORT'
                    
                    logger.info(f"Opportunity found: {symbol} | Score: {score:.2f} | {side}")
                    
                    # Send preview signal
                    preview_msg = (
                        f"🔍 <b>Potential Trade Detected</b>\n\n"
                        f"<b>Pair:</b> {symbol}\n"
                        f"<b>Direction:</b> {side}\n"
                        f"<b>Confidence Score:</b> {score:.2f}/1.0\n\n"
                        f"<i>Evaluating entry...</i>"
                    )
                    self.send_telegram_alert(preview_msg)
                    
                    return symbol, direction, position_size
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        return None, None, None

    def execute_trade(self, symbol: str, direction: int, position_size: float) -> bool:
        """Execute trade with proper risk management and send Telegram alert"""
        logger.info(f"Attempting {symbol} trade with size: {position_size:.2f} USDT")
        
        if direction == 1:  # Long
            profit = (ticker['last'] - price) * qty
        else:  # Short
            profit = (price - ticker['last']) * qty

        self.total_profit += profit
        self.trade_count += 1
        logger.info(f"Cumulative P&L: ${self.total_profit:.2f} | Trades: {self.trade_count}")
        
        # Get current price with retries
        price = None
        for _ in range(3):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker['last'])
                if price > 0:
                    break
            except:
                time.sleep(1)
        
        if not price:
            logger.error(f"Could not get valid price for {symbol}")
            return False
            
        # Check minimum order size
        if (position_size * self.leverage) < self.min_order_size:
            logger.warning(f"Position size too small after leverage: {(position_size * self.leverage):.2f} USDT")
            return False
            
        try:
            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol)
            
            # Calculate position quantity
            market = self.exchange.market(symbol)
            qty = (position_size * self.leverage) / price
            qty = round(qty, market['precision']['amount'])
            
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
                    'stopLoss': str(sl_price),
                    'takeProfit': str(tp_price),
                    'positionIdx': 0
                }
                
            )

            # Send trade execution alert
            alert_msg = (
                f"🚀 <b>New Trade Signal</b> 🚀\n\n"
                f"<b>Pair:</b> {symbol}\n"
                f"<b>Direction:</b> {side.upper()}\n"
                f"<b>Entry:</b> {price:.4f}\n"
                f"<b>Stop Loss:</b> {sl_price:.4f}\n"
                f"<b>Take Profit:</b> {tp_price:.4f}\n"
                f"<b>Size:</b> {position_size:.2f} USDT\n"
                f"<b>Leverage:</b> {self.leverage}x\n\n"
                f"<i>Automated trade executed</i>"
            )
            self.send_telegram_alert(alert_msg)
            
            logger.info(f"Trade executed: {symbol} | {side.upper()} | Qty: {qty:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            self.send_telegram_alert(f"❌ Trade failed: {symbol} - {str(e)}")
            return False

    def run(self):
        """Main trading loop"""
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
    try:
        bot = BybitTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        sys.exit(1)
