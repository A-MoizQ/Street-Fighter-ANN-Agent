class Buttons:
    def __init__(self, buttons_dict=None):
        if buttons_dict is not None:
            self.dict_to_object(buttons_dict)
        else:
            self.init_buttons()

    def init_buttons(self):
        self.up = False
        self.down = False
        self.right = False
        self.left = False
        self.select = False
        self.start = False
        self.Y = False
        self.B = False
        self.X = False
        self.A = False
        self.L = False
        self.R = False

    def dict_to_object(self, buttons_dict):
        # Normalize incoming keys to uppercase for consistency
        bd = {k.upper(): v for k, v in buttons_dict.items()}
        self.up = bd.get('UP', False)
        self.down = bd.get('DOWN', False)
        self.right = bd.get('RIGHT', False)
        self.left = bd.get('LEFT', False)
        self.select = bd.get('SELECT', False)
        self.start = bd.get('START', False)
        self.Y = bd.get('Y', False)
        self.B = bd.get('B', False)
        self.X = bd.get('X', False)
        self.A = bd.get('A', False)
        self.L = bd.get('L', False)
        self.R = bd.get('R', False)

    def object_to_dict(self):
        return {
            'UP': self.up,
            'DOWN': self.down,
            'RIGHT': self.right,
            'LEFT': self.left,
            'SELECT': self.select,
            'START': self.start,
            'Y': self.Y,
            'B': self.B,
            'X': self.X,
            'A': self.A,
            'L': self.L,
            'R': self.R,
        }