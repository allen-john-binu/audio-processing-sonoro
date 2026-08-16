import os
import sys
import csv
import socket
import subprocess
import threading
from datetime import datetime, timedelta

import numpy as np
import scipy.signal as signal
import sounddevice as sd
from flask import Flask, request, jsonify, render_template_string


# ============================================================
# Configuration
# ============================================================

FS = 48000

DURN = 5e-3                 # 5 ms chirp
WAIT_TIME = 0.1             # 100 ms
TARGET_DURATION = 300       # 5 minutes

CHIRP_START = 2000
CHIRP_END = 24000
AMPLITUDE = 0.8

BASE_LEFT = 1.0
BASE_RIGHT = 1.0

CSV_SAMPLE_INTERVAL = 1920  # 10 ms at 48 kHz

PORT = 5007


# ============================================================
# Output directories
# ============================================================

OUTPUT_DIR = "LabRealWithCam/dataFromSound/"
EMPTY_COPY_DIR = "LabRealWithCam/dataFromReal/"
EMPTY_COPY_DIR_1 = "LabRealWithCam/dataFromVicon/"
EMPTY_COPY_DIR_2 = "LabRealWithCam/dataFromRing/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMPTY_COPY_DIR, exist_ok=True)
os.makedirs(EMPTY_COPY_DIR_1, exist_ok=True)
os.makedirs(EMPTY_COPY_DIR_2, exist_ok=True)


# ============================================================
# Generate the sound
# Exactly the same sound generation as your original script
# ============================================================

def generate_sound():

    t = np.linspace(
        0,
        DURN,
        int(FS * DURN),
        endpoint=False
    )

    sweep = signal.chirp(
        t,
        CHIRP_START,
        t[-1],
        CHIRP_END
    )

    sweep *= signal.windows.tukey(
        len(sweep),
        0.2
    )

    sweep *= AMPLITUDE

    sweep = np.pad(
        sweep,
        (
            int(FS * WAIT_TIME),
            int(FS * WAIT_TIME)
        )
    )

    repeats = int(
        np.ceil(
            TARGET_DURATION /
            (len(sweep) / FS)
        )
    )

    long_signal = np.tile(
        sweep,
        repeats
    )

    long_signal = long_signal[
        :int(FS * TARGET_DURATION)
    ]

    return long_signal


print("Generating sound...")
long_signal = generate_sound()
print(
    f"Sound ready: {len(long_signal) / FS:.2f} seconds"
)


# ============================================================
# Constant channel volumes
# ============================================================

left_volumes = np.full(
    len(long_signal),
    BASE_LEFT
)

right_volumes = np.full(
    len(long_signal),
    BASE_RIGHT
)


stereo_signal = np.column_stack((
    long_signal * left_volumes,
    long_signal * right_volumes
))


# ============================================================
# Experiment state
# ============================================================

state_lock = threading.Lock()

experiment_name = None
csv_rows = []

sample_index = 0
start_wall_time = None

stream = None

state = "ready"

save_lock = threading.Lock()


# ============================================================
# File handling
# ============================================================

def get_filenames(name):

    if not name.lower().endswith(".csv"):
        name += ".csv"

    return {
        "sound": os.path.join(
            OUTPUT_DIR,
            name
        ),

        "real": os.path.join(
            EMPTY_COPY_DIR,
            name
        ),

        "vicon": os.path.join(
            EMPTY_COPY_DIR_1,
            name
        ),

        "ring": os.path.join(
            EMPTY_COPY_DIR_2,
            name
        )
    }


def files_already_exist(name):

    filenames = get_filenames(name)

    return any(
        os.path.exists(path)
        for path in filenames.values()
    )


# ============================================================
# Save CSV files
# ============================================================

def save_experiment():

    global csv_rows
    global experiment_name

    with save_lock:

        if experiment_name is None:
            return

        filenames = get_filenames(
            experiment_name
        )

        print("\nWriting CSV...")

        # Main sound CSV
        with open(
            filenames["sound"],
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "sample_index",
                "time_seconds",
                "left_volume",
                "right_volume"
            ])

            writer.writerows(csv_rows)

        # Empty CSV files
        open(
            filenames["real"],
            "w"
        ).close()

        open(
            filenames["vicon"],
            "w"
        ).close()

        open(
            filenames["ring"],
            "w"
        ).close()

        print(
            f"CSV saved: {filenames['sound']}"
        )

        print(
            f"Empty CSV created:"
        )

        print(
            f"  {filenames['real']}"
        )

        print(
            f"  {filenames['vicon']}"
        )

        print(
            f"  {filenames['ring']}"
        )

        print(
            f"Total rows: {len(csv_rows)}"
        )


# ============================================================
# Audio callback
# ============================================================

def audio_callback(
    outdata,
    frames,
    time,
    status
):

    global sample_index
    global start_wall_time
    global state

    if status:
        print(status)

    chunk = stereo_signal[
        sample_index:
        sample_index + frames
    ]

    if len(chunk) < frames:

        if len(chunk) > 0:
            outdata[:len(chunk)] = chunk

        outdata[len(chunk):] = 0

        # Experiment reached 300 seconds
        with state_lock:
            state = "finished"

        raise sd.CallbackStop()

    else:

        outdata[:] = chunk


    # --------------------------------------------------------
    # Initialize wall-clock reference
    # Same timing approach as original script
    # --------------------------------------------------------

    if start_wall_time is None:

        start_wall_time = (
            datetime.now()
            -
            timedelta(
                seconds=time.outputBufferDacTime
            )
        )


    # --------------------------------------------------------
    # Log samples
    # Same CSV timing approach as original script
    # --------------------------------------------------------

    for i in range(
        0,
        frames,
        CSV_SAMPLE_INTERVAL
    ):

        idx = sample_index + i

        if idx >= len(stereo_signal):
            break

        dac_time = (
            time.outputBufferDacTime
            +
            (i / FS)
        )

        wall_time = (
            start_wall_time
            +
            timedelta(seconds=dac_time)
        )

        csv_rows.append([
            wall_time.isoformat(),
            idx,
            f"{idx / FS:.6f}",
            f"{left_volumes[idx]:.6f}",
            f"{right_volumes[idx]:.6f}"
        ])


    sample_index += frames


# ============================================================
# Start experiment
# ============================================================

def start_experiment(name):

    global experiment_name
    global csv_rows
    global sample_index
    global start_wall_time
    global stream
    global state

    with state_lock:

        if state == "playing":
            return False, "An experiment is already running."

        if not name:
            return False, "Experiment name is required."

        name = name.strip()

        if not name:
            return False, "Experiment name is required."

        if files_already_exist(name):
            return False, (
                "Files with this experiment name already exist."
            )

        # ----------------------------------------------
        # Reset experiment state
        # ----------------------------------------------

        experiment_name = name

        csv_rows = []

        sample_index = 0

        start_wall_time = None

        state = "playing"

        # ----------------------------------------------
        # Start audio
        # ----------------------------------------------

        try:

            stream = sd.OutputStream(
                samplerate=FS,
                channels=2,
                callback=audio_callback,
                blocksize=0
            )

            stream.start()

        except Exception as e:

            stream = None
            state = "ready"
            experiment_name = None

            return False, str(e)

    print()
    print("=" * 60)
    print(f"EXPERIMENT STARTED: {name}")
    print("=" * 60)

    return True, "Experiment started."


# ============================================================
# Stop experiment
# ============================================================

def stop_experiment():

    global stream
    global state
    global experiment_name

    with state_lock:

        if state != "playing":
            return False, "No experiment is currently running."

        current_stream = stream

        # Mark stopped immediately
        state = "saving"


    # --------------------------------------------------------
    # Stop audio immediately
    # --------------------------------------------------------

    if current_stream is not None:

        try:
            current_stream.stop()
        except Exception:
            pass

        try:
            current_stream.close()
        except Exception:
            pass


    stream = None


    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    save_experiment()


    with state_lock:
        state = "ready"


    print()
    print("=" * 60)
    print("EXPERIMENT STOPPED")
    print("=" * 60)
    print()


    return True, "Experiment stopped and saved."


# ============================================================
# Automatically finish after 300 seconds
# ============================================================

def finish_experiment():

    global stream
    global state

    with state_lock:

        if state != "finished":
            return

        current_stream = stream


    if current_stream is not None:

        try:
            current_stream.stop()
        except Exception:
            pass

        try:
            current_stream.close()
        except Exception:
            pass


    stream = None


    print()
    print("=" * 60)
    print("300 SECONDS REACHED")
    print("Saving experiment...")
    print("=" * 60)


    save_experiment()


    with state_lock:
        state = "ready"


    print("Ready for next experiment.")
    print()


# ============================================================
# Watch for natural completion
# ============================================================

def completion_watcher():

    while True:

        with state_lock:
            current_state = state

        if current_state == "finished":
            finish_experiment()

        threading.Event().wait(0.1)


# Start watcher
threading.Thread(
    target=completion_watcher,
    daemon=True
).start()


# ============================================================
# Find Mac Wi-Fi IP
# ============================================================

def get_local_ip():

    # macOS Wi-Fi interface is usually en0.
    # Sometimes it can be en1.

    for interface in ["en0", "en1"]:

        try:

            result = subprocess.check_output(
                [
                    "ipconfig",
                    "getifaddr",
                    interface
                ],
                stderr=subprocess.DEVNULL
            )

            ip = result.decode().strip()

            if ip:
                return ip

        except Exception:
            pass


    # Fallback
    try:

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        s.connect(
            ("8.8.8.8", 80)
        )

        ip = s.getsockname()[0]

        s.close()

        return ip

    except Exception:

        return "127.0.0.1"


# ============================================================
# Flask web application
# ============================================================

app = Flask(__name__)


HTML = """
<!DOCTYPE html>

<html>

<head>

    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <title>Sound Experiment</title>

    <style>

        body {
            font-family: -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;

            background: #f4f4f4;

            margin: 0;
            padding: 30px 20px;

            text-align: center;
        }

        .container {
            max-width: 500px;
            margin: auto;
            background: white;

            padding: 30px 20px;

            border-radius: 16px;

            box-shadow:
                0 4px 20px rgba(0,0,0,0.10);
        }

        h1 {
            margin-top: 0;
            font-size: 28px;
        }

        .status {
            font-size: 20px;
            font-weight: 600;

            margin: 25px 0;
        }

        input {
            box-sizing: border-box;

            width: 100%;

            font-size: 20px;

            padding: 15px;

            border: 1px solid #ccc;

            border-radius: 10px;

            margin-bottom: 20px;
        }

        button {
            width: 100%;

            font-size: 22px;

            font-weight: 600;

            padding: 17px;

            border: none;

            border-radius: 10px;

            margin-top: 10px;

            cursor: pointer;
        }

        #startButton {
            background: #2e7d32;
            color: white;
        }

        #stopButton {
            background: #c62828;
            color: white;
        }

        button:disabled {
            background: #ccc !important;
            color: #777 !important;
        }

        .message {
            margin-top: 20px;

            font-size: 16px;

            min-height: 24px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>Sound Experiment</h1>

    <div class="status">
        Status:
        <span id="status">Ready</span>
    </div>


    <input
        id="experimentName"
        type="text"
        placeholder="Experiment name"
        autocomplete="off"
    >


    <button
        id="startButton"
        onclick="startExperiment()"
    >
        START
    </button>


    <button
        id="stopButton"
        onclick="stopExperiment()"
        disabled
    >
        STOP
    </button>


    <div
        id="message"
        class="message"
    ></div>

</div>


<script>

let statusTimer = null;


function updateUI(status, name) {

    const statusElement =
        document.getElementById("status");

    const startButton =
        document.getElementById("startButton");

    const stopButton =
        document.getElementById("stopButton");

    const input =
        document.getElementById("experimentName");


    if (status === "playing") {

        statusElement.textContent =
            "Playing";

        startButton.disabled = true;

        stopButton.disabled = false;

        input.disabled = true;

    }

    else {

        statusElement.textContent =
            "Ready";

        startButton.disabled = false;

        stopButton.disabled = true;

        input.disabled = false;

        if (name) {
            input.value = "";
        }

    }

}


async function startExperiment() {

    const input =
        document.getElementById("experimentName");

    const message =
        document.getElementById("message");

    const name =
        input.value.trim();


    if (!name) {

        message.textContent =
            "Please enter an experiment name.";

        return;
    }


    message.textContent =
        "Starting...";


    try {

        const response =
            await fetch("/start", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    name: name
                })
            });


        const data =
            await response.json();


        if (!data.success) {

            message.textContent =
                data.message;

            return;
        }


        message.textContent =
            data.message;

        updateUI("playing", name);

    }

    catch (error) {

        message.textContent =
            "Could not connect to the server.";

    }

}


async function stopExperiment() {

    const message =
        document.getElementById("message");


    message.textContent =
        "Stopping and saving...";


    try {

        const response =
            await fetch("/stop", {

                method: "POST"

            });


        const data =
            await response.json();


        if (!data.success) {

            message.textContent =
                data.message;

            return;
        }


        message.textContent =
            data.message;

        updateUI("ready", true);

    }

    catch (error) {

        message.textContent =
            "Could not connect to the server.";

    }

}


async function checkStatus() {

    try {

        const response =
            await fetch("/status");

        const data =
            await response.json();


        if (data.state === "playing") {

            updateUI(
                "playing",
                data.experiment_name
            );

        }

        else {

            updateUI(
                "ready",
                false
            );

        }

    }

    catch (error) {

        // Ignore temporary connection errors

    }

}


// Keep phone display synchronized
statusTimer = setInterval(
    checkStatus,
    1000
);


checkStatus();

</script>

</body>

</html>
"""


# ============================================================
# Web routes
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


@app.route(
    "/status",
    methods=["GET"]
)
def get_status():

    with state_lock:

        return jsonify({
            "state": state,
            "experiment_name": experiment_name
        })


@app.route(
    "/start",
    methods=["POST"]
)
def web_start():

    data = request.get_json(
        silent=True
    )

    if not data:
        return jsonify({
            "success": False,
            "message": "Invalid request."
        })


    name = data.get(
        "name",
        ""
    )


    success, message = start_experiment(
        name
    )


    return jsonify({
        "success": success,
        "message": message
    })


@app.route(
    "/stop",
    methods=["POST"]
)
def web_stop():

    success, message = stop_experiment()


    return jsonify({
        "success": success,
        "message": message
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    local_ip = get_local_ip()


    print()
    print("=" * 60)
    print("SOUND EXPERIMENT SERVER")
    print("=" * 60)

    print()
    print("Open this address on your phone:")
    print()
    print(
        f"    http://{local_ip}:{PORT}"
    )

    print()
    print("The phone and Mac must be on the same Wi-Fi.")
    print()
    print("Press Ctrl+C on the Mac to quit the server.")
    print()
    print("=" * 60)
    print()


    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True
    )