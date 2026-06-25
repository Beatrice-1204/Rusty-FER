import asyncio
import threading
import tkinter as tk
from bleak import BleakClient

# -------- BLE CONFIG --------
TARGET_ADDRESS = "DA:AA:0D:6C:E3:A5"
CHAR_UUID = "FFE1"

# -------- BLE STATE --------
ble_client = None
loop = asyncio.new_event_loop()

# -------- BLE FUNCTIONS --------
async def ble_connect():
    global ble_client
    ble_client = BleakClient(TARGET_ADDRESS)
    await ble_client.connect()
    print("Conectat la HC-42")

async def ble_send(cmd):
    if ble_client and ble_client.is_connected:
        await ble_client.write_gatt_char(CHAR_UUID, cmd.encode(), response=False)

def start_ble():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_connect())
    loop.run_forever()

threading.Thread(target=start_ble, daemon=True).start()

# -------- GUI --------
root = tk.Tk()
root.title("Robot Mecanum Control (HC-42)")
root.geometry("420x640")
root.resizable(False, False)

speed_val = tk.IntVar(value=5)

def send(cmd):
    asyncio.run_coroutine_threadsafe(ble_send(cmd), loop)

def set_speed(val):
    send(str(val))

# -------- BUTTONS --------
btn_cfg = {"font": ("Arial", 16), "width": 6, "height": 2}

tk.Button(root, text="↑", command=lambda: send("w"), **btn_cfg).place(x=160, y=40)
tk.Button(root, text="↓", command=lambda: send("s"), **btn_cfg).place(x=160, y=140)
tk.Button(root, text="←", command=lambda: send("a"), **btn_cfg).place(x=80,  y=90)
tk.Button(root, text="→", command=lambda: send("d"), **btn_cfg).place(x=240, y=90)

tk.Button(root, text="⟲", command=lambda: send("q"), **btn_cfg).place(x=80,  y=200)
tk.Button(root, text="⟳", command=lambda: send("e"), **btn_cfg).place(x=240, y=200)

tk.Button(
    root, text="STOP", bg="red", fg="white",
    font=("Arial", 18), width=10, height=2,
    command=lambda: send("x")
).place(x=100, y=270)

# -------- SPEED SLIDER --------
tk.Label(root, text="VITEZĂ", font=("Arial", 14)).place(x=170, y=360)

tk.Scale(
    root, from_=1, to=9, orient=tk.HORIZONTAL,
    length=300, variable=speed_val,
    command=set_speed
).place(x=60, y=400)

# -------- EMOTION TEST BUTTONS --------
tk.Label(root, text="REACTII", font=("Arial", 14)).place(x=165, y=480)

tk.Button(root, text="Happy", command=lambda: send("H"), width=8, height=2).place(x=45, y=520)
tk.Button(root, text="Angry", command=lambda: send("G"), width=8, height=2).place(x=135, y=520)
tk.Button(root, text="Sad", command=lambda: send("V"), width=8, height=2).place(x=225, y=520)
tk.Button(root, text="Surprise", command=lambda: send("U"), width=8, height=2).place(x=315, y=520)

root.mainloop()
