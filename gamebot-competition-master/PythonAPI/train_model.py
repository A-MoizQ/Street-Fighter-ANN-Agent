import os, re, pickle
import pandas as pd, numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Reshape, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Player 1 buttons only
BUTTONS = ['up', 'down', 'right', 'left', 'select', 'start', 'y', 'b', 'x', 'a', 'l', 'r']
P1_BUTTON_COLS = [f'player1_buttons_{b}' for b in BUTTONS]

def calculate_class_weights(df, cols):
    ws, n = [], len(df)
    for c in cols:
        pos = df[c].sum()
        neg = n - pos
        w = (neg / pos) if pos > 0 else 10.0
        ws.append(min(w, 10.0))
    return tf.constant(ws, dtype=tf.float32)

def create_weighted_loss(df, cols):
    pos_w = calculate_class_weights(df, cols)  # (12,)
    bce_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    def loss(y_true, y_pred):
        # Compute element-wise binary cross-entropy, shape=(batch_size, 12)
        bce = bce_fn(y_true, y_pred)
        
        # Broadcast pos_w to match the shape of y_true and y_pred, shape=(batch_size, 12)
        w = tf.where(y_true == 1, pos_w, 1.0)
        
        # Apply weights and compute the mean loss
        weighted_bce = bce * w
        return tf.reduce_mean(weighted_bce)
    
    return loss

def build_model(input_dim):
    return Sequential([
        Reshape((6, input_dim // 6), input_shape=(input_dim,)),
        Bidirectional(LSTM(256, return_sequences=True)), Dropout(0.3),
        Bidirectional(LSTM(128)), Dropout(0.3),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(len(P1_BUTTON_COLS), activation='sigmoid')
    ])

def train_model(csv_path, model_path):
    df = pd.read_csv(csv_path)
    # 1) Map all fight_result cols (including t-lags)
    fr_map = {'NOT_OVER': 0, 'P1': 1, 'P2': 2, 'DRAW': 3}
    for c in df.columns:
        if c.startswith('fight_result'):
            df[c] = df[c].map(fr_map).fillna(0).astype(int)
    # 2) Convert bool → int
    for c in df.select_dtypes(include='bool').columns:
        df[c] = df[c].astype(int)
    # 3) Drop or convert any other object columns
    for c in df.select_dtypes(include='object').columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            df.drop(columns=[c], inplace=True)
    df.fillna(0, inplace=True)

    # targets & features
    y_cols = [c for c in df.columns if c in P1_BUTTON_COLS]
    X_cols = [c for c in df.columns if c not in y_cols]

    # normalize numeric, non‐binary features
    pat = re.compile(r'_buttons_|_jumping|_crouching|_in_move|has_round_started|is_round_over|fight_result|_id', re.I)
    num_cols = [c for c in X_cols
                if df[c].dtype in [np.int64, np.float64]
                and not pat.search(c)
                and df[c].nunique() > 2]
    if num_cols:
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        with open(model_path.replace('.keras', '_scaler.pkl'), 'wb') as f:
            pickle.dump({'cols': num_cols, 'scaler': scaler}, f)

    # split
    X = df[X_cols].values.astype(np.float32)
    y = df[y_cols].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # build & compile
    model = build_model(X_tr.shape[1])
    loss_fn = create_weighted_loss(df, y_cols)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # callbacks
    cbs = [
        EarlyStopping('val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, 'val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    # train
    model.fit(X_tr, y_tr,
              validation_data=(X_te, y_te),
              epochs=50, batch_size=64,
              callbacks=cbs, verbose=1)

    # eval & save
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"Test loss: {loss:.5f}, accuracy: {acc:.4f}")
    model.save(model_path)

if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flattened_window_datasets'))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(out, exist_ok=True)
    for fn in os.listdir(base):
        if fn.startswith('windowed_dataset_') and fn.endswith('.csv'):
            cid = fn.split('_')[-1].split('.')[0]
            inp = os.path.join(base, fn)
            mdl = os.path.join(out, f'model_{cid}.keras')
            print(f"\n=== Training character {cid} ===")
            train_model(inp, mdl)