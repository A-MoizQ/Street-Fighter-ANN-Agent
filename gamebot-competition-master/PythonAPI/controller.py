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
MODE = 'record' if len(sys.argv) > 2 and sys.argv[2] == 'record' else 'bot'
port = 9999 if player_id == '1' else 10000

def connect(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', port))
    s.listen(1)
    client, _ = s.accept()
    print('Connected to game!')
    return client

def send(sock, cmd):
    sock.sendall(json.dumps(cmd.object_to_dict()).encode())

def receive(sock):
    data = json.loads(sock.recv(4096).decode())
    return GameState(data)

# Main loop
def main():
    sock = connect(port)
    cmd = Command()
    from bot import Bot
    bot = Bot()

    while True:
        gs = receive(sock)
        if MODE == 'record':
            keys = get_current_keypress()
            record_frame(gs, keys)
            # Forward human input
            cmd.player_buttons = Buttons({k: True for k in keys})
            send(sock, cmd)
        else:
            cmd = bot.fight(gs, player_id)
            send(sock, cmd)
        time.sleep(1/60.0)

if __name__ == '__main__':
    main()