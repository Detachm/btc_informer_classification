import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class BitcoinOptimizedFeatureEngineer:
    """比特币优化特征工程器 - 选择最重要的特征"""
    
    def __init__(self):
        self.feature_names = []
        
    def add_core_features(self, df):
        """核心价格特征"""
        features = []
        
        # 基础OHLCV
        features.extend(['open', 'high', 'low', 'close', 'volume'])
        
        # 关键价格比率
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features.extend(['hl_ratio', 'oc_ratio', 'price_position'])
        
        return features
    
    def add_essential_technical_indicators(self, df):
        """核心技术指标"""
        features = []
        
        # 关键移动平均线
        for period in [5, 10, 20]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            features.append(f'ma_{period}')
        
        # MACD指标
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        features.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # RSI指标
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        features.append('rsi_14')
        
        # 布林带
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'])
        
        return features
    
    def add_volatility_features(self, df):
        """波动率特征"""
        features = []
        
        # 历史波动率
        for period in [5, 20]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
            features.append(f'volatility_{period}')
        
        # ATR
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        features.append('atr')
        
        return features
    
    def add_momentum_features(self, df):
        """动量特征"""
        features = []
        
        # 价格动量
        for period in [3, 10]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            features.append(f'momentum_{period}')
        
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        features.append('price_change')
        
        return features
    
    def add_volume_features(self, df):
        """成交量特征"""
        features = []
        
        # 成交量移动平均
        for period in [5, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            features.append(f'volume_ma_{period}')
        
        # 成交量比率
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        features.append('volume_ratio')
        
        return features
    
    def add_pattern_features(self, df):
        """形态特征"""
        features = []
        
        # 支撑阻力位
        for period in [10, 20]:
            df[f'resistance_{period}'] = df['high'].rolling(window=period).max()
            df[f'support_{period}'] = df['low'].rolling(window=period).min()
            df[f'resistance_distance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
            df[f'support_distance_{period}'] = (df['close'] - df[f'support_{period}']) / df['close']
            features.extend([f'resistance_distance_{period}', f'support_distance_{period}'])
        
        return features
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices, period=20, std=2):
        """计算布林带"""
        ma = prices.rolling(window=period).mean()
        std_val = prices.rolling(window=period).std()
        upper = ma + (std_val * std)
        lower = ma - (std_val * std)
        return upper, ma, lower
    
    def calculate_atr(self, high, low, close, period=14):
        """计算ATR指标"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def create_optimized_features(self, df):
        """创建优化的特征集"""
        print("开始创建优化特征...")
        
        # 核心特征
        core_features = self.add_core_features(df)
        print(f"✓ 核心特征: {len(core_features)} 个")
        
        # 技术指标
        tech_features = self.add_essential_technical_indicators(df)
        print(f"✓ 技术指标: {len(tech_features)} 个")
        
        # 波动率特征
        vol_features = self.add_volatility_features(df)
        print(f"✓ 波动率特征: {len(vol_features)} 个")
        
        # 动量特征
        momentum_features = self.add_momentum_features(df)
        print(f"✓ 动量特征: {len(momentum_features)} 个")
        
        # 成交量特征
        volume_features = self.add_volume_features(df)
        print(f"✓ 成交量特征: {len(volume_features)} 个")
        
        # 形态特征
        pattern_features = self.add_pattern_features(df)
        print(f"✓ 形态特征: {len(pattern_features)} 个")
        
        # 合并所有特征
        all_features = (core_features + tech_features + vol_features + 
                       momentum_features + volume_features + pattern_features)
        
        self.feature_names = all_features
        print(f"\n总特征数量: {len(all_features)} 个")
        
        return all_features

def test_optimized_features():
    """测试优化特征工程"""
    print("测试优化特征工程...")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    test_data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1min'),
        'open': 50000 + np.random.randn(n_samples) * 1000,
        'high': 50000 + np.random.randn(n_samples) * 1000 + 500,
        'low': 50000 + np.random.randn(n_samples) * 1000 - 500,
        'close': 50000 + np.random.randn(n_samples) * 1000,
        'volume': np.random.randint(100, 1000, n_samples)
    }
    
    df = pd.DataFrame(test_data)
    
    # 创建特征工程师
    engineer = BitcoinOptimizedFeatureEngineer()
    features = engineer.create_optimized_features(df)
    
    print(f"\n特征列表:")
    for i, feature in enumerate(features):
        print(f"  {i+1:2d}. {feature}")
    
    return engineer, features

if __name__ == "__main__":
    engineer, features = test_optimized_features()
