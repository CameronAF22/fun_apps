import pickle
import pathlib

class PatchedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        return super().find_class(module, name)

# Expose the custom unpickler as Unpickler
Unpickler = PatchedUnpickler

def load(file):
    with open(file, "rb") as f:
        return Unpickler(f).load()