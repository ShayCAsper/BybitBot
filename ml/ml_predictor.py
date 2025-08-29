"""
Machine Learning Prediction System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

class MLPredictor:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_weights = {
            'xgboost': 0.3,
            'lstm': 0.3,
            'random_forest': 0.2,
            'ensemble': 0.2
        }
        
    async def load_models(self):
        """Load pre-trained models"""
        try:
            # In production, load actual trained models
            # For now, we'll initialize placeholder models
            logger.info("Loading ML models...")
            
            # Initialize models (in production, load from files)
            self.models['xgboost'] = self.create_xgboost_model()
            self.models['random_forest'] = self.create_rf_model()
            # LSTM would be loaded from TensorFlow/Keras
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
    
    def create_xgboost_model(self):
        """Create XGBoost model (placeholder)"""
        # In production, this would load a trained model
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=100, max_depth=5)
    
    def create_rf_model(self):
        """Create Random Forest model (placeholder)"""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100)
    
    async def predict(self, market_data: Dict) -> Dict:
        """Generate predictions for all symbols"""
        predictions = {}
        
        for symbol, data in market_data.items():
            try:
                features = self.extract_features(data)
                if features is not None:
                    prediction = self.make_prediction(features)
                    predictions[symbol] = prediction
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
        
        return predictions
    
    def extract_features(self, data: Dict) -> Optional[np.ndarray]:
        """Extract features from market data"""
        try:
            ohlcv = data.get('ohlcv', []) 
            if len(ohlcv) < 50:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate technical features
            features = []
            
            # Price features
            features.append(df['close'].pct_change().iloc[-1])
            features.append((df['close'].iloc[-1] - df['close'].mean()) / df['close'].std())
            features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])
            
            # Volume features
            
            last_volume = df['volume'].iloc[-1] if not df['volume'].empty else np.nan
            mean_volume = df['volume'].mean()

            # Safe division (avoid nan/inf)
            if pd.notna(last_volume) and pd.notna(mean_volume) and mean_volume != 0:
                features.append(last_volume / mean_volume)
            else:
                features.append(0)  # or np.nan depending on how you want to handle it

            # Safe pct_change
            if len(df['volume']) > 1 and pd.notna(df['volume'].iloc[-1]):
                pct = df['volume'].pct_change().iloc[-1]
                features.append(pct if pd.notna(pct) else 0)
            else:
                features.append(0)
            #features.append(df['volume'].iloc[-1] / df['volume'].mean())
            #features.append(df['volume'].pct_change().iloc[-1])
            
            
            
            
            # Moving averages
            features.append(df['close'].iloc[-1] / df['close'].rolling(10).mean().iloc[-1])
            features.append(df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1])
            
            
            
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1])
            
            # Volatility
            features.append(df['close'].pct_change().std())
            
            # Order book features (if available)
            orderbook = data.get('orderbook', {})
            if orderbook:
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                if bids and asks:
                    bid_volume = sum([b[1] for b in bids[:10]])
                    ask_volume = sum([a[1] for a in asks[:10]])
                    if bid_volume + ask_volume > 0:
                        features.append((bid_volume - ask_volume) / (bid_volume + ask_volume))
                    else:
                        features.append(0)
                else:
                    features.append(0)
            else:
                features.append(0)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def make_prediction(self, features: np.ndarray) -> Dict:
        """Make prediction using ensemble of models"""
        predictions = []
        confidences = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    # For demo, generate random predictions
                    # In production, use actual model predictions
                    pred = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                    conf = np.random.uniform(0.5, 0.9)
                    
                    predictions.append(pred * self.model_weights.get(model_name, 0.25))
                    confidences.append(conf * self.model_weights.get(model_name, 0.25))
                    
                except Exception as e:
                    logger.error(f"Error in {model_name} prediction: {e}")
        
        # Ensemble prediction
        if predictions:
            final_prediction = np.sum(predictions)
            final_confidence = np.mean(confidences)
        else:
            final_prediction = 0
            final_confidence = 0.5
        
        return {
            'signal': final_prediction,  # -1: sell, 0: neutral, 1: buy
            'confidence': final_confidence,
            'timestamp': datetime.now()
        }