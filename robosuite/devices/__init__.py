from .device import Device
from .keyboard import Keyboard

try:
    from .spacemouse import SpaceMouse
except ImportError:
    print(
        """Unable to load module hid, required to interface with SpaceMouse.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
try:
    from .gamepad import GamePad
except ImportError:
    print(
        """Unable to load module hid, required to interface with GamePad. \n
        Install the additional requirements with \n
        `pip install -r requirements-extra.txt`"""
    )
