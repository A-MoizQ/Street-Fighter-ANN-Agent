import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from command import Command
from buttons import Buttons
import time

# Define constants the same way as done when training
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over',
                 'player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id',
                 'player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
                 'diff_x', 'diff_y', 'diff_health']

BUTTONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'Y', 'B', 'X', 'A', 'L', 'R']
P1_BUTTON_COLS = [f'player1_buttons_{b.lower()}' for b in BUTTONS]

class Bot:
    def __init__(self, player_id=0, model_path=None):
        self.buttons = Buttons()
        self.cmd = Command()
        self.player_id = player_id
        
        # locate model & scaler
        if model_path is None:
            # Use RNN_models directory instead of models
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RNN_models'))
            model_path = os.path.join(base, f'model_{player_id}.keras')
        
        # Load model and scaler
        print(f"[RNN Bot] Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        scaler_path = model_path + '.scaler'
        self.scaler = joblib.load(scaler_path)

        # 🎯 CONVERT SCALER PARAMETERS TO TENSORFLOW TENSORS
        # This allows us to include scaling in the compiled function
        self.scaler_mean_ = tf.constant(self.scaler.mean_, dtype=tf.float32)
        self.scaler_scale_ = tf.constant(self.scaler.scale_, dtype=tf.float32)
        
        print(f"[RNN Bot] Converting scaler to TensorFlow tensors...")
        print(f"[RNN Bot] Scaler mean shape: {self.scaler_mean_.shape}")
        print(f"[RNN Bot] Scaler scale shape: {self.scaler_scale_.shape}")

        # Init frame buffer
        self.buffer = deque(maxlen=WINDOW_SIZE)
        empty = {feat: 0 for feat in STATE_FEATURES}
        empty['fight_result'] = 'NOT_OVER'
        for _ in range(WINDOW_SIZE):
            self.buffer.append(empty.copy())

        # 🎯 CREATE COMPILED INFERENCE FUNCTION
        print(f"[RNN Bot] Compiling inference function with @tf.function...")
        self._compiled_predict = self._create_compiled_predict_function()
        
        # 🎯 WARM UP THE COMPILED FUNCTION
        print(f"[RNN Bot] Warming up compiled function...")
        dummy_input = tf.random.normal((1, WINDOW_SIZE, len(STATE_FEATURES)), dtype=tf.float32)
        
        # Run multiple warmup calls to ensure compilation
        for i in range(5):
            _ = self._compiled_predict(dummy_input)
        
        print(f"[RNN Bot] ✅ Model loaded and compiled successfully!")
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0

    def _create_compiled_predict_function(self):
        """Create a compiled TensorFlow function for ultra-fast inference"""
        
        @tf.function(
            experimental_relax_shapes=True,  # Allow variable input shapes if needed
            jit_compile=True,  # Enable XLA compilation for even more speed
            input_signature=[tf.TensorSpec(shape=[1, WINDOW_SIZE, len(STATE_FEATURES)], dtype=tf.float32)]
        )
        def compiled_predict(X_3d):
            """
            Compiled prediction function that includes:
            1. Scaling (using TF tensors instead of sklearn)
            2. Model prediction
            3. All in one compiled graph for maximum speed
            """
            
            # 🎯 FAST SCALING USING TENSORFLOW OPERATIONS
            # Reshape to 2D for scaling
            batch_size = tf.shape(X_3d)[0]
            X_reshaped = tf.reshape(X_3d, [-1, len(STATE_FEATURES)])
            
            # Apply scaling: (X - mean) / scale
            X_scaled = (X_reshaped - self.scaler_mean_) / self.scaler_scale_
            
            # Reshape back to 3D
            X_scaled_3d = tf.reshape(X_scaled, [batch_size, WINDOW_SIZE, len(STATE_FEATURES)])
            
            # 🎯 MODEL PREDICTION
            predictions = self.model(X_scaled_3d, training=False)
            
            return predictions
        
        return compiled_predict

    def _frame_to_dict(self, gs):
        """Convert GameState to feature dictionary (optimized)"""
        p1, p2 = gs.player1, gs.player2
        d = {
            'timer': gs.timer,
            'fight_result': gs.fight_result,
            'has_round_started': int(gs.has_round_started),
            'is_round_over': int(gs.is_round_over),
            'player1_id': p1.player_id,
            'p1_health': p1.health,
            'p1_x': p1.x_coord,
            'p1_y': p1.y_coord,
            'p1_jumping': int(p1.is_jumping),
            'p1_crouching': int(p1.is_crouching),
            'p1_in_move': int(p1.is_player_in_move),
            'p1_move_id': p1.move_id,
            'player2_id': p2.player_id,
            'p2_health': p2.health,
            'p2_x': p2.x_coord,
            'p2_y': p2.y_coord,
            'p2_jumping': int(p2.is_jumping),
            'p2_crouching': int(p2.is_crouching),
            'p2_in_move': int(p2.is_player_in_move),
            'p2_move_id': p2.move_id,
            'diff_x': p1.x_coord - p2.x_coord,
            'diff_y': p1.y_coord - p2.y_coord,
            'diff_health': p1.health - p2.health,
        }
        return d

    def _prepare_tensor_input(self):
        """Prepare input as TensorFlow tensor for compiled function"""
        X_3d = []
        FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}

        # Create a sequence for each time step
        for t in range(WINDOW_SIZE):
            frame = self.buffer[t]
            # Extract features for this frame
            features = []
            for feat in STATE_FEATURES:
                val = frame[feat]
                if feat == 'fight_result':
                    features.append(FIGHT_MAP.get(val, 0))
                else:
                    features.append(float(val))
            X_3d.append(features)
        
        # Convert to TensorFlow tensor with shape (1, WINDOW_SIZE, n_features)
        X_tensor = tf.constant([X_3d], dtype=tf.float32)
        return X_tensor

    def fight(self, gs, player_id):
        """Optimized fight function using compiled TensorFlow operations"""
        
        import time
        start_time = time.perf_counter()
        
        # 1. Append new frame
        raw = self._frame_to_dict(gs)
        self.buffer.append(raw)

        # 2. Prepare input as TensorFlow tensor
        X_tensor = self._prepare_tensor_input()
        
        # 3. 🎯 USE COMPILED PREDICTION FUNCTION (MUCH FASTER!)
        predictions = self._compiled_predict(X_tensor)
        
        # Convert to numpy for further processing
        preds = predictions.numpy()[0]
        
        # Track performance
        inference_time = time.perf_counter() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Report performance every 50 inferences
        if self.inference_count % 50 == 0:
            avg_time = self.total_inference_time / self.inference_count
            print(f"\n[RNN Bot] 📊 Performance Update:")
            print(f"  Inference #{self.inference_count}")
            print(f"  Current inference: {inference_time*1000:.1f}ms")
            print(f"  Average inference: {avg_time*1000:.1f}ms")
            print(f"  Estimated FPS capability: {1/avg_time:.0f}")
        
        # 4. Process predictions (show only significant probabilities)
        if self.inference_count % 100 == 0:  # Show details less frequently
            print(f"\n[RNN Bot] Prediction probabilities for Player {player_id}:")
            for button, prob in zip(BUTTONS, preds):
                if prob > 0.05:  # Only show buttons with >5% probability
                    print(f"  {button}: {prob:.1%}")

        # 5. Map to Buttons, resolving opposing directions
        probs = {b: float(preds[i]) for i, b in enumerate(BUTTONS)}
        
        # 🎯 OPTIMIZED CONFLICT RESOLUTION
        # Remove pairs that cancel each other out
        if probs['LEFT'] > 0.01 and probs['RIGHT'] > 0.01:
            # Pick the stronger
            if probs['LEFT'] > probs['RIGHT']:
                probs['RIGHT'] = 0.0
            else:
                probs['LEFT'] = 0.0
        
        if probs['UP'] > 0.01 and probs['DOWN'] > 0.01:
            if probs['UP'] > probs['DOWN']:
                probs['DOWN'] = 0.0
            else:
                probs['UP'] = 0.0

        # Final button map with higher threshold for cleaner actions
        threshold = 0.3  # Increased threshold for more decisive actions
        btn_map = {b: (probs[b] > threshold) for b in BUTTONS}
        
        cmd = Command()
        if str(player_id) == "1":
            cmd.player_buttons = Buttons(btn_map)
        else:
            cmd.player2_buttons = Buttons(btn_map)
        
        # Show active buttons only occasionally to reduce spam
        if self.inference_count % 30 == 0:
            active_buttons = [btn for btn, state in btn_map.items() if state]
            print(f"[RNN Bot] Active buttons for Player {player_id}: {', '.join(active_buttons) if active_buttons else 'None'}")

        return cmd

    def get_performance_stats(self):
        """Get performance statistics"""
        if self.inference_count == 0:
            return None
        
        avg_inference_time = self.total_inference_time / self.inference_count
        return {
            'inference_count': self.inference_count,
            'avg_inference_time': avg_inference_time,
            'estimated_fps': 1 / avg_inference_time if avg_inference_time > 0 else 0,
            'total_time': self.total_inference_time
        }

# 🎯 PERFORMANCE TEST FUNCTION
def benchmark_compiled_rnn(player_id=0, iterations=100):
    """Benchmark the compiled RNN bot performance"""
    
    print(f"🚀 Benchmarking Compiled RNN Bot (Player {player_id})")
    print(f"Running {iterations} iterations...")
    
    try:
        # Initialize bot
        bot = Bot(player_id=player_id)
        
        # Create dummy game state
        class DummyGameState:
            def __init__(self):
                self.timer = 100
                self.fight_result = 'NOT_OVER'
                self.has_round_started = True
                self.is_round_over = False
                self.player1 = DummyPlayer(0)
                self.player2 = DummyPlayer(1)
        
        class DummyPlayer:
            def __init__(self, pid):
                self.player_id = pid
                self.health = 100
                self.x_coord = 50.0
                self.y_coord = 0.0
                self.is_jumping = False
                self.is_crouching = False
                self.is_player_in_move = False
                self.move_id = 0
        
        dummy_gs = DummyGameState()
        
        # Warm up
        print("Warming up...")
        for _ in range(10):
            _ = bot.fight(dummy_gs, "1")
        
        # Benchmark
        print(f"Running benchmark...")
        start_time = time.perf_counter()
        
        for i in range(iterations):
            cmd = bot.fight(dummy_gs, "1")
        
        total_time = time.perf_counter() - start_time
        avg_time = total_time / iterations
        
        print(f"\n✅ Benchmark Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per inference: {avg_time*1000:.1f}ms")
        print(f"  Potential FPS: {1/avg_time:.0f}")
        print(f"  Performance improvement target: <5ms per inference")
        
        stats = bot.get_performance_stats()
        if stats:
            print(f"  Bot internal stats: {stats['avg_inference_time']*1000:.1f}ms avg")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Run performance test
    benchmark_compiled_rnn(player_id=0, iterations=100)