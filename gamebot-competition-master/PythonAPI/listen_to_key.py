from pynput import keyboard

_pressed = set()

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

def on_press(key):
    try:
        _pressed.add(key.char.upper())
    except AttributeError:
        _pressed.add(str(key).upper())

def on_release(key):
    try:
        _pressed.discard(key.char.upper())
    except AttributeError:
        _pressed.discard(str(key).upper())

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

def get_current_keypress():
    mapped = set()
    for k in _pressed:
        if k in custom_keymap:
            mapped.add(custom_keymap[k])
    return list(mapped)