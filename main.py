import os
from enum import Enum
from pathlib import Path
from datetime import datetime
import time

from pycromanager import Core, Studio
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

syncboard_main_serial = SerialConnection("/dev/syncboard", 115200)
syncboard_magnet_serial = SerialConnection("/dev/magnet", 115200)

syncboard_main = SyncBoardController(syncboard_main_serial)
syncboard_magnet = SyncBoardController(syncboard_magnet_serial)

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
    img = mmc.snap_image()
    liveManager = studio.getSnapLiveManager()
    liveManager.displayImage(img)

    # Set up the syncboards
    syncboard_main.attach_leds()
    syncboard_main.enable_system()
    syncboard_main._setup_leds()

    syncboard_magnet.attach_magnet()
    syncboard_magnet.enable_system()
    syncboard_magnet.setup_magnet()

    # Calibrate the magnet
    syncboard_magnet.calibrate_magnet()
    input("Move the Hall sensor to the sample position, then press enter to continue calibration...")
    syncboard_magnet.calibrate_hall()

    input("Now move back to the sample, then press enter to start the experiment...")
    syncboard_main.enable_led(LED_IDS.LED_450_NM, led_intensity)

    for i in range(10):
        syncboard_magnet.set_magnet_field(5 * i / 10)
        mmc.snap_image()
        time.sleep(image_interval_s)

    syncboard_magnet.set_magnet_field(0)

def main():
    try:
        run_experiment()
    finally:
        print("Disabling syncboards.")
        syncboard_main.disable_all_leds()
        syncboard_main.disable_system()
        syncboard_magnet.disable_system()

if __name__ == '__main__':
    main()
