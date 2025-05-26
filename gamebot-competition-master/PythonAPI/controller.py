import socket
import json
import sys
import time
from game_state import GameState
from command import Command
from buttons import Buttons
from listen_to_key import get_current_keypress
from make_dataset import record_frame

player_id = sys.argv[1]
MODE = 'record' if len(sys.argv) > 2 and sys.argv[2] == 'record' else \
       'train_dqn' if len(sys.argv) > 2 and sys.argv[2] == 'train_dqn' else 'bot'

MODEL_TYPE = 'dqn' if len(sys.argv) > 2 and sys.argv[2] == 'dqn' else \
            'rnn' if len(sys.argv) > 2 and sys.argv[2] == 'rnn' else 'standard'

port = 9999 if player_id == '1' else 10000

def connect(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', port))
    s.listen(1)
    client, _ = s.accept()
    print('Connected to game!')
    return client

def send(sock, cmd):
    payload = json.dumps(cmd.object_to_dict())
    print("[Controller] Sending:", payload)
    sock.sendall(payload.encode())

def receive(sock):
    data = json.loads(sock.recv(4096).decode())
    return GameState(data)

def main():
    sock = connect(port)
    cmd = Command()
    player_id_set = False
    bot = None
    trainer = None
    
    while True:
        gs = receive(sock)
        
        if not player_id_set:
            if MODE == 'train_dqn':
                from dqn_trainer import DQNTrainer
                trainer = DQNTrainer(player_id=gs.player1.player_id)
                print("[Controller] DQN Training mode activated")
            elif MODE != 'record':
                if MODEL_TYPE == 'dqn':
                    from dqn_bot import DQNBot as Bot
                    print("[Controller] Using DQN reinforcement learning bot")
                elif MODEL_TYPE == 'rnn':
                    from rnn_bot import Bot
                    print("[Controller] Using RNN model bot")
                else:
                    from bot import Bot
                    print("[Controller] Using standard model bot")
                    
                bot = Bot(player_id=gs.player1.player_id)
            player_id_set = True

        if MODE == 'record':
            keys = get_current_keypress()
            cmd.player_buttons = Buttons({k: True for k in keys})
            record_frame(gs, keys)
        elif MODE == 'train_dqn':
            cmd = trainer.train_step(gs, player_id)
        else:
            cmd = bot.fight(gs, player_id)
        
        send(sock, cmd)
        time.sleep(1/60.0)
        
if __name__ == '__main__':
    main()