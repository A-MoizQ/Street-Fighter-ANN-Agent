import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from command import Command
from buttons import Buttons

#define constants the same way as done when training
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over','player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id','player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id','diff_x', 'diff_y', 'diff_health']
#making feature columns to feed into ANN just like during the training time
FEATURE_COLS = []
for t in range(WINDOW_SIZE-1, -1, -1):
    suffix = f"_t-{t}"
    for feat in STATE_FEATURES:
        FEATURE_COLS.append(feat + suffix)
BUTTONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'Y', 'B', 'X', 'A', 'L', 'R']
P1_BUTTON_COLS = [f'player1_buttons_{b}' for b in BUTTONS]

class Bot:
    def __init__(self,player_id=0, model_path=None):
        self.buttons = Buttons()
        self.cmd = Command()
        # locate model & scaler
        if model_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
            model_path = os.path.join(base, 'model_'+f'{player_id}' +'.keras')
        #load model and scaler
        self.model = tf.keras.models.load_model(model_path)
        scaler_path = model_path + '.scaler'
        self.scaler = joblib.load(scaler_path)

        # init frame buffer
        self.buffer = deque(maxlen=WINDOW_SIZE)
        empty = {feat: 0 for feat in STATE_FEATURES}
        empty['fight_result'] = 'NOT_OVER'
        for _ in range(WINDOW_SIZE):
            self.buffer.append(empty.copy())

    def _frame_to_dict(self, gs):
        # map GameState to raw feature dict (no suffix)
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

    def fight(self, gs, player_id):
        # 1. append new frame
        # print("[Bot] Incoming GameState raw data:", gs.__dict__)
        raw = self._frame_to_dict(gs)
        # print("[Bot] Mapped frame:", raw)
        self.buffer.append(raw)
        # print("[Bot] Buffer contents:", list(self.buffer))

        # 2. build flattened feature list
        flat = []
        FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}
        for t in range(WINDOW_SIZE-1, -1, -1):
            frame = self.buffer[WINDOW_SIZE-1 - t]
            for feat in STATE_FEATURES:
                val = frame[feat]
                if feat == 'fight_result':
                    flat.append(FIGHT_MAP[val])
                else:
                    flat.append(int(val))

        # 3. create DataFrame then scale
        df_feat = pd.DataFrame([flat], columns=FEATURE_COLS)
        X_scaled = self.scaler.transform(df_feat)
        # print(X_scaled)

        # 4. predict
        preds = self.model.predict(X_scaled, verbose=0)[0]
        # print("Current Predictions: ", preds)
        print("\nPrediction probabilities for each button:")
        for button, prob in zip(BUTTONS, preds):
            if prob > 0.005:  # Only show buttons with >0.5% probability
                print(f"{button}: {prob:.2%}")

        # 5. map to Buttons
        # btn_map = {b: bool(preds[i] > 0.25) for i, b in enumerate(BUTTONS)}
        # print("Mapped Buttons: ", btn_map)
        # 5. map to Buttons, but resolve opposing directions:
        probs = {b: float(preds[i]) for i, b in enumerate(BUTTONS)}
        #remove pairs that cancel each other out
        if probs['LEFT'] and probs['RIGHT']:
            # pick the stronger
            if probs['LEFT'] > probs['RIGHT']:
                probs['RIGHT'] = 0.0
            else:
                probs['LEFT'] = 0.0
        if probs['UP'] and probs['DOWN']:
            if probs['UP'] > probs['DOWN']:
                probs['DOWN'] = 0.0
            else:
                probs['UP'] = 0.0

        #final button map
        btn_map = {b: (probs[b] > 0.005) for b in BUTTONS}
        
        cmd = Command()
        if player_id == "1":
            cmd.player_buttons = Buttons(btn_map)
            print("[Bot Debug] Button map:", btn_map)
            print("[Bot Debug] Command buttons state:", cmd.player_buttons.__dict__)
        else:
            cmd.player2_buttons = Buttons(btn_map)
        
        # print(f"[Bot] Sending command with predictions: {cmd.object_to_dict()}")
        active_buttons = [btn for btn, state in btn_map.items() if state]
        print(f"\nActive buttons for Player {player_id}:", ", ".join(active_buttons) if active_buttons else "None")

        return cmd