import os
from enum import Enum
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt

from pycromanager import Core, Studio

# Add sync_board folder to path
import sys
sys.path.append("/home/hslab/Documents/Gabi/MagnetExperiment/sync_board")

from sync_board.syncboard.syncboardcontroller import SyncBoardController
from sync_board.syncboard.serialconnection import SerialConnection

from asitiger.tigercontroller import TigerController

from dmd_socket import DMDControl

led_intensity = 0.1
image_interval_s = 10

if led_intensity > 0.29:
    raise ValueError("LED intensity must be less than 0.29 for CW measurements.")

class LED_IDS(Enum):
    NO_LED = -1
    LED_385_NM = 0
    LED_450_NM = 1
    LED_515_NM = 2
    LED_565_NM = 3
    LED_645_NM = 4

    # OLD LEDs
    LED_405_NM = 5
    LED_505_NM = 6
    LED_538_NM = 7

class FILTER_WHEEL(Enum):
    FILTER = 0, 
    BLOCKING = 1, 
    NO_FILTER = 2

TIGER_CARD_ADDR_FW = 8

date = datetime.now().strftime("%Y-%m-%d")
save_dir = Path(f"/media/hslab/Data/ImageData/Gabi/{date}/MARY/")
if not save_dir.exists():
    os.makedirs(save_dir)

syncboard_serial = SerialConnection("/dev/syncboard", 2000000)
syncboard = SyncBoardController(syncboard_serial)

# $ pip install asitiger
# Make sure filter wheel is set to the correct position
tiger = TigerController.from_serial_port("/dev/ttyUSB0")
tiger.filter_wheel(position=FILTER_WHEEL.FILTER.value, card_address=TIGER_CARD_ADDR_FW)

# Set the DMD to full white
dmd = DMDControl()
dmd.initialise()
if not dmd.is_initialised():
    raise RuntimeError("DMD not initialised.")
dmd.display_full()

# Set up imaging
mmc = Core()
studio = Studio()

def run_experiment():
    
    # Set up the syncboards
    syncboard.attach_leds()
    syncboard.attach_magnet()
    syncboard.enable_system()
    syncboard._setup_leds()
    syncboard.setup_magnet()

    # Calibrate the magnet
    syncboard.calibrate_magnet()
    input("Move the Hall sensor to the sample position, then press enter to continue calibration...")
    # syncboard.calibrate_hall(0)

    input("Now move back to the sample, then press enter to start the experiment...")
    # syncboard.enable_led(LED_IDS.LED_450_NM.value, led_intensity)
    
    while (led_intensity := input("Enter LED intensity (0-1) or c to continue: ")) != "c":
        led_intensity = float(led_intensity)
        if led_intensity > 1.0:
            print("LED intensity must be between 0 and 1.")
            continue
        syncboard.enable_led(LED_IDS.LED_450_NM.value, led_intensity)
    
    syncboard.set_magnet_current(0)
    syncboard.enable_magnet()

    i = 0
    while True:
        print(f"Iteration {i}")
        if i % 2 == 0:
            set_current = 0
        else:
            set_current = 0.5
        print(f"Setting current to {set_current} A")
        syncboard.set_magnet_current(set_current)
        # field = syncboard.read_hall(0)
        # print(f"Measured Field: {field} mT")
        i += 1
        time.sleep(image_interval_s)

    syncboard.set_magnet_current(0)
    syncboard.enable_magnet(False)

def main():
    try:
        # run_experiment()
        syncboard.attach_leds()
        syncboard.enable_system()
        syncboard._setup_leds()
        time.sleep(1)
        syncboard.enable_led(LED_IDS.LED_450_NM.value, led_intensity, 1000)
        time.sleep(1)
    finally:
        print("Disabling syncboards.")
        syncboard.disable_all_leds()
        syncboard.disable_system()

if __name__ == '__main__':
    main()
