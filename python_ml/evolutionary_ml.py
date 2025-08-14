import asyncio
import numpy as np
import pandas as pd
import json
import sys
import time
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StrategyGene:
    momentum_threshold: float
    reversion_threshold: float
    volume_threshold: float
    confidence_multiplier: float
    position_size_factor: float
    hold_time_factor: float
    
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, input_size=50, hidden_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.quantum_layer = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.Sigmoid(),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # [price_direction, confidence, hold_time]
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        quantum = self.quantum_layer(encoded)
        prediction = self.predictor(quantum)
        return prediction

class EvolutionaryMLSystem:
    def __init__(self):
        self.device = self._get_device()
        self.neural_networks = {}
        self.strategy_population = []
        self.performance_history = []
        self.generation = 0
        self.session = None
        
        # Initialize multiple neural networks for different strategies
        self.init_neural_networks()
        
        # Initialize genetic algorithm population
        self.init_strategy_population()
        
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def init_neural_networks(self):
        strategies = ['momentum', 'reversion', 'volume', 'scalping', 'arbitrage']
        
        for strategy in strategies:
            self.neural_networks[strategy] = {
                'model': QuantumNeuralNetwork().to(self.device),
                'optimizer': None,
                'scaler': StandardScaler(),
                'performance': 0.0,
                'trades_count': 0
            }
            
            self.neural_networks[strategy]['optimizer'] = optim.AdamW(
                self.neural_networks[strategy]['model'].parameters(),
                lr=0.001, weight_decay=0.01
            )
    
    def init_strategy_population(self, population_size=50):
        self.strategy_population = []
        
        for _ in range(population_size):
            gene = StrategyGene(
                momentum_threshold=np.random.uniform(1.0, 10.0),
                reversion_threshold=np.random.uniform(5.0, 25.0),
                volume_threshold=np.random.uniform(50000, 500000),
                confidence_multiplier=np.random.uniform(0.5, 2.0),
                position_size_factor=np.random.uniform(0.5, 2.0),
                hold_time_factor=np.random.uniform(0.1, 3.0)
            )
            self.strategy_population.append({
                'gene': gene,
                'fitness': 0.0,
                'trades': 0,
                'pnl': 0.0
            })
    
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def fetch_market_data(self) -> Dict:
        session = await self.get_session()
        
        try:
            async with session.get("https://www.okx.com/api/v5/market/tickers?instType=SPOT") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
        
        return {}
    
    def extract_features(self, market_data: Dict) -> np.ndarray:
        if not market_data.get('data'):
            return np.array([])
        
        features = []
        
        for ticker in market_data['data'][:50]:  # Top 50 by volume
            try:
                if not ticker['instId'].endswith('-USDT'):
                    continue
                    
                price = float(ticker['last'])
                volume = float(ticker.get('vol24h', 0))
                
                if price <= 0 or volume <= 0:
                    continue
                
                # Calculate features
                bid = float(ticker.get('bidPx', price))
                ask = float(ticker.get('askPx', price))
                spread = (ask - bid) / bid * 10000 if bid > 0 else 0
                
                # Add normalized features
                features.extend([
                    np.log(price + 1),
                    np.log(volume + 1),
                    spread,
                    np.tanh(spread / 10.0),  # Normalized spread
                ])
                
            except (ValueError, KeyError):
                continue
        
        # Pad or truncate to fixed size
        if len(features) < 50:
            features.extend([0.0] * (50 - len(features)))
        else:
            features = features[:50]
        
        return np.array(features, dtype=np.float32)
    
    async def generate_predictions(self, features: np.ndarray) -> Dict[str, float]:
        predictions = {}
        
        if len(features) == 0:
            return predictions
        
        # Reshape for batch processing
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        for strategy_name, strategy_data in self.neural_networks.items():
            model = strategy_data['model']
            model.eval()
            
            with torch.no_grad():
                prediction = model(features_tensor)
                
                # Extract prediction components
                direction = torch.tanh(prediction[0, 0]).item()  # -1 to 1
                confidence = torch.sigmoid(prediction[0, 1]).item()  # 0 to 1
                hold_time = torch.sigmoid(prediction[0, 2]).item()  # 0 to 1
                
                predictions[strategy_name] = {
                    'direction': direction,
                    'confidence': confidence,
                    'hold_time': hold_time,
                    'signal_strength': abs(direction) * confidence
                }
        
        return predictions
    
    def evolve_strategies(self, performance_data: Dict):
        # Update fitness based on performance
        for i, individual in enumerate(self.strategy_population):
            # Use recent performance as fitness
            individual['fitness'] = performance_data.get('total_pnl', 0.0) + \
                                  performance_data.get('win_rate', 0.0) * 100.0
        
        # Sort by fitness
        self.strategy_population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Keep top 50% as parents
        parents = self.strategy_population[:len(self.strategy_population)//2]
        
        # Generate new population
        new_population = parents.copy()  # Keep elites
        
        while len(new_population) < len(self.strategy_population):
            # Select two parents
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            
            # Crossover
            child_gene = StrategyGene(
                momentum_threshold=(parent1['gene'].momentum_threshold + parent2['gene'].momentum_threshold) / 2,
                reversion_threshold=(parent1['gene'].reversion_threshold + parent2['gene'].reversion_threshold) / 2,
                volume_threshold=(parent1['gene'].volume_threshold + parent2['gene'].volume_threshold) / 2,
                confidence_multiplier=(parent1['gene'].confidence_multiplier + parent2['gene'].confidence_multiplier) / 2,
                position_size_factor=(parent1['gene'].position_size_factor + parent2['gene'].position_size_factor) / 2,
                hold_time_factor=(parent1['gene'].hold_time_factor + parent2['gene'].hold_time_factor) / 2,
            )
            
            # Mutation
            if np.random.random() < 0.1:  # 10% mutation rate
                child_gene.momentum_threshold *= np.random.uniform(0.8, 1.2)
                child_gene.reversion_threshold *= np.random.uniform(0.8, 1.2)
                child_gene.volume_threshold *= np.random.uniform(0.8, 1.2)
                child_gene.confidence_multiplier *= np.random.uniform(0.8, 1.2)
                child_gene.position_size_factor *= np.random.uniform(0.8, 1.2)
                child_gene.hold_time_factor *= np.random.uniform(0.8, 1.2)
            
            new_population.append({
                'gene': child_gene,
                'fitness': 0.0,
                'trades': 0,
                'pnl': 0.0
            })
        
        self.strategy_population = new_population
        self.generation += 1
        
        logger.info(f"Evolution complete - Generation {self.generation}")
    
    def get_best_strategy_weights(self) -> Dict[str, float]:
        if not self.strategy_population:
            return {
                'MOMENTUM': 1.0,
                'SCALPING': 1.0,
                'REVERSION': 1.0,
                'VOLUME': 1.0,
                'ML_PRED': 1.0
            }
        
        # Get best performing individual
        best_individual = max(self.strategy_population, key=lambda x: x['fitness'])
        gene = best_individual['gene']
        
        # Convert genetic parameters to strategy weights
        weights = {
            'MOMENTUM': gene.confidence_multiplier * gene.momentum_threshold / 10.0,
            'SCALPING': gene.position_size_factor * 1.5,
            'REVERSION': gene.reversion_threshold / 20.0,
            'VOLUME': gene.volume_threshold / 100000.0,
            'ML_PRED': gene.hold_time_factor * 2.0
        }
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total * 5.0 for k, v in weights.items()}  # Scale to reasonable range
        
        return weights
    
    async def train_neural_networks(self, market_data: Dict):
        # Generate training data from market movements
        if not market_data.get('data'):
            return
        
        features = self.extract_features(market_data)
        if len(features) == 0:
            return
        
        # Simulate training with synthetic targets based on market conditions
        for strategy_name, strategy_data in self.neural_networks.items():
            model = strategy_data['model']
            optimizer = strategy_data['optimizer']
            
            # Create synthetic targets based on strategy type
            if strategy_name == 'momentum':
                # Target should predict strong directional moves
                target = torch.FloatTensor([[0.5, 0.8, 0.3]]).to(self.device)  # [direction, confidence, hold_time]
            elif strategy_name == 'reversion':
                # Target should predict mean reversion opportunities
                target = torch.FloatTensor([[-0.3, 0.7, 0.6]]).to(self.device)
            else:
                # Default target
                target = torch.FloatTensor([[0.0, 0.5, 0.5]]).to(self.device)
            
            # Forward pass
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prediction = model(features_tensor)
            
            # Calculate loss
            loss = nn.MSELoss()(prediction, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            strategy_data['trades_count'] += 1
    
    async def continuous_evolution(self):
        logger.info("🧠 Starting continuous ML evolution...")
        
        while True:
            try:
                # Fetch latest market data
                market_data = await self.fetch_market_data()
                
                if market_data:
                    # Train neural networks
                    await self.train_neural_networks(market_data)
                    
                    # Generate predictions
                    features = self.extract_features(market_data)
                    predictions = await self.generate_predictions(features)
                    
                    # Simulate performance data (in real system, this would come from trading engine)
                    performance_data = {
                        'total_pnl': np.random.normal(0, 1),  # Simulated P&L
                        'win_rate': np.random.uniform(0.4, 0.8),
                        'total_trades': self.generation * 10
                    }
                    
                    # Evolve strategies
                    self.evolve_strategies(performance_data)
                    
                    # Output results for Rust engine
                    evolution_results = {
                        'generation': self.generation,
                        'strategy_weights': self.get_best_strategy_weights(),
                        'confidence_threshold': 0.6 + np.random.random() * 0.3,
                        'neural_predictions': predictions,
                        'timestamp': time.time()
                    }
                    
                    print(json.dumps(evolution_results))
                    
                await asyncio.sleep(30)  # Evolve every 30 seconds
                
            except Exception as e:
                logger.error(f"Evolution error: {e}")
                await asyncio.sleep(5)
    
    async def close(self):
        if self.session:
            await self.session.close()

async def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--evolve':
        ml_system = EvolutionaryMLSystem()
        
        try:
            # Single evolution cycle for testing
            market_data = await ml_system.fetch_market_data()
            if market_data:
                await ml_system.train_neural_networks(market_data)
                
                features = ml_system.extract_features(market_data)
                predictions = await ml_system.generate_predictions(features)
                
                performance_data = {'total_pnl': 0.0, 'win_rate': 0.5, 'total_trades': 0}
                ml_system.evolve_strategies(performance_data)
                
                results = {
                    'generation': ml_system.generation,
                    'strategy_weights': ml_system.get_best_strategy_weights(),
                    'confidence_threshold': 0.7,
                    'neural_predictions': predictions
                }
                
                print(json.dumps(results))
        finally:
            await ml_system.close()
    else:
        # Continuous evolution mode
        ml_system = EvolutionaryMLSystem()
        try:
            await ml_system.continuous_evolution()
        finally:
            await ml_system.close()

if __name__ == "__main__":
    asyncio.run(main())
