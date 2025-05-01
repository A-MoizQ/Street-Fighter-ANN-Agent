import socket
import json
import sys
import time
from game_state import GameState
from command import Command
from buttons import Buttons
from listen_to_key import get_current_keypress
from make_dataset import record_frame

# Mode switch: 'record' to capture human input, default runs bot
player_id = sys.argv[1]
MODE = 'record' if len(sys.argv) > 2 and sys.argv[2] == 'record' else 'bot'
port = 9999 if player_id == '1' else 10000


def connect(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', port))
    server_socket.listen(1)
    client_socket, _ = server_socket.accept()
    print('Connected to game!')
    return client_socket


def send(client_socket, cmd):
    client_socket.sendall(json.dumps(cmd.object_to_dict()).encode())


def receive(client_socket):
    payload = client_socket.recv(4096)
    data = json.loads(payload.decode())
    return GameState(data)


def main():
    client_socket = connect(port)
    cmd = Command()

    while True:
        gs = receive(client_socket)

        if MODE == 'record':
            # Log dataset row
            record_frame(gs)

            # Forward your keypresses to the game
            keys = get_current_keypress()
            # Inside controller.py, after keys = get_current_keypress()
            custom_keymap = {
                'A': 'Y',
                'S': 'B',
                'D': 'X',
                'Z': 'A',
                'X': 'L',
                'C': 'R',
                'UP': 'UP',
                'DOWN': 'DOWN',
                'LEFT': 'LEFT',
                'RIGHT': 'RIGHT',
                'ENTER': 'START',
                'SPACE': 'SELECT'
            }

            buttons_dict = {custom_keymap[k]: True for k in keys if k in custom_keymap}
            new_buttons = Buttons(buttons_dict)
            cmd.player_buttons = new_buttons

            send(client_socket, cmd)

        else:
            # Run rule-based bot
            from bot import Bot
            bot = Bot()
            cmd = bot.fight(gs, player_id)
            send(client_socket, cmd)

        time.sleep(1/60.0)

if __name__ == '__main__':
    main()