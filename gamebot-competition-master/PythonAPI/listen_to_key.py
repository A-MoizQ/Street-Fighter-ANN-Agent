from pynput import keyboard

_pressed = set()

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

# Start listener in background
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

def get_current_keypress():
    """
    Returns a list of currently pressed keys (e.g. ['LEFT', 'A']).
    """
    return list(_pressed)