# main_integration.py
import os  # For operating system path operations and file checks
import time  # For sleep delays between commands\ nimport torch  # PyTorch library for tensors and model handling
from torch.utils.data import DataLoader  # To create data loaders from datasets
from ScentRealmForNeckWear.ScentRealmProtocol import NeckWear  # Import the NeckWear SDK class
import serial  # PySerial for serial communication with the device
import serial.tools.list_ports  # To list available serial ports

# Import the custom dataset and multimodal model
from models.dataset import SmellReproductionDataset
from models.model import MultimodalOdorNet

def handshake_device(ser, nw):
    """
    Perform getUuid and wakeUp handshake with the NeckWear device.
    """
    # 1. Request the device UUID channel
    uuid_cmd = nw.getUuid()  # Ask the SDK to generate the getUuid hex command
    ser.write(bytes.fromhex(uuid_cmd.replace(" ", "")))  # Send the command over serial
    time.sleep(1)  # Wait 1 second for the device to respond
    resp = ser.read_all()  # Read all available bytes from serial
    if resp:
        # Convert received bytes to space-separated hex string
        hex_resp = " ".join(f"{b:02X}" for b in resp)
        cmd, channel = nw.cmdParse(hex_resp)  # Use SDK to parse response
        print(f"Handshake: device channel = {channel}")  # Log the channel
    # 2. Wake up the device repeatedly to ensure it is active
    wake_cmd = nw.wakeUp()  # Generate the wakeUp command
    for _ in range(5):
        ser.write(bytes.fromhex(wake_cmd.replace(" ", "")))  # Send wakeUp
        time.sleep(0.5)  # Brief pause between commands
    print("Handshake: device awakened")  # Log completion

def main():
    # 1. Configuration parameters
    DATA_ROOT      = "smell_data"  # Directory with sensor CSV subfolders
    IMAGE_ROOT     = "images"  # Directory with image subfolders
    DESC_PATH      = "smell_descriptions.json"  # JSON file with text descriptions
    CHECKPOINT     = "models/checkpoint.pt"  # Path to saved model weights
    BATCH_SIZE     = 1  # Number of samples per batch in inference
    SENSOR_SEQ_LEN = 100  # Fixed length for sensor sequences
    DURATION       = 8  # Seconds to emit each scent
    DEFAULT_PORT   = "COM3"  # Default serial port name

    # 2. Serial port setup
    ports = [str(p).split('-')[0].strip() for p in serial.tools.list_ports.comports()]
    print("Available serial ports:", ports)  # List ports
    port = input(f"Select serial port [{DEFAULT_PORT}]: ") or DEFAULT_PORT  # User selects port
    try:
        ser = serial.Serial(port=port, baudrate=115200, timeout=1)  # Open serial connection
        print(f"Opened serial port: {ser.port}")  # Confirm opened
    except Exception as e:
        print("Error opening serial port:", e)  # Print error if fails
        return  # Exit main

    # 3. Handshake sequence
    nw = NeckWear()  # Instantiate the SDK handler
    handshake_device(ser, nw)  # Perform getUuid + wakeUp

    # 4. Prepare dataset and data loader
    dataset = SmellReproductionDataset(
        sensor_root=DATA_ROOT,
        image_root=IMAGE_ROOT,
        desc_path=DESC_PATH,
        sensor_seq_len=SENSOR_SEQ_LEN
    )  # Instantiate dataset
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # Create loader

    # 5. Load trained model
    device = torch.device("cpu")  # Use CPU for inference
    model = MultimodalOdorNet().to(device)  # Instantiate and move to device
    try:
        state = torch.load(CHECKPOINT, map_location=device)  # Load weights
        model.load_state_dict(state)  # Apply weights
        model.eval()  # Switch to evaluation mode
        print("Model loaded and set to eval mode")  # Confirm load
    except Exception as e:
        print("Error loading model:", e)  # Report errors
        ser.close()  # Close serial on failure
        return  # Exit main

    # 6. Inference and odor emission loop
    try:
        for batch in loader:
            sensor = batch["sensor"].to(device)  # Tensor shape [B, L, F]
            image  = batch["image"]  # PIL.Image instance
            text   = batch["text"]  # List of description strings

            with torch.no_grad():
                mix_ratios = model(image, text, sensor)  # Forward pass

            probs = mix_ratios.squeeze(0)  # Remove batch dimension => [num_perfumes]
            scent_id = int(probs.argmax().item()) + 1  # Select top scent (+1 for 1-based)
            print(f"Predicted scent ID: {scent_id}")  # Log prediction

            cmd_hex = nw.playSmell(scent_id, DURATION)  # Generate playSmell command
            packet = bytes.fromhex(cmd_hex.replace(" ", ""))  # Convert to bytes
            ser.write(packet)  # Send to device
            print(f"Sent PlaySmell command: {cmd_hex}")  # Log command

            time.sleep(DURATION + 1)  # Wait for emission
    finally:
        # 7. Cleanup actions always executed
        stop_cmd = nw.stopPlay()  # Generate stop command
        ser.write(bytes.fromhex(stop_cmd.replace(" ", "")))  # Send stop
        print(f"Sent StopPlay command: {stop_cmd}")  # Log stop
        ser.close()  # Close serial port
        print("Serial port closed")  # Confirm closure

if __name__ == "__main__":
    main()  # Run main when script is executed
