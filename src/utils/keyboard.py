import threading
from pynput import keyboard

def resolve_key(name):
    if hasattr(keyboard.Key, name): return getattr(keyboard.Key, name)
    if name.startswith("f") and name[1:].isdigit(): return getattr(keyboard.Key, name, None)
    if len(name) == 1: return keyboard.KeyCode.from_char(name)
    return None

class KeyListener:
    def __init__(self):
        self.state = {}; self.pressed_edges = set(); self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self.listener.start()

    def _press(self, key):
        with self.lock:
            if not self.state.get(key, False): self.pressed_edges.add(key)
            self.state[key] = True

    def _release(self, key):
        with self.lock: self.state[key] = False

    def take_edge(self, name):
        k = resolve_key(name)
        if k is None: return False
        with self.lock:
            if k in self.pressed_edges: 
                self.pressed_edges.discard(k)
                return True
        return False

    def is_pressed(self, name):
        k = resolve_key(name)
        if k is None: return False
        with self.lock: return self.state.get(k, False)

    def stop(self): 
        self.listener.stop()