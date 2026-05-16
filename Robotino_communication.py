# Importing necessary libraries
import socket
import requests
# Robotino's connection info
IP_ADDRESS = '192.168.0.1'  # Local Robotino IP address
PORT = 80  # Port to connect to

# Connect to Robotino
def connect_to_robotino():
    try:
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((IP_ADDRESS, PORT))
        print("Successfully connected to Robotino!")
        return sock
    except Exception as e:
        print(f"Error connecting to Robotino: {e}")
        return None
# Get raw odometry from robotino
def get_odometry():
    try:
        # Send HTTP GET request to retrieve proximity sensor values
        url = f"http://{IP_ADDRESS}/data/odometry"
        response = requests.get(url)
        # Check if the response is successful
        if response.status_code == 200:
            # Assuming the response returns an array of floats in a JSON format
            odometry_readings = response.json()  # Parse JSON response
            if len(odometry_readings) == 7:
                return odometry_readings
            else:
                print("Unexpected number of sensor values received.")
        else:
            print(f"Error: Received status code {response.status_code}")
    except Exception as e:
        print(f"Error retrieving sensor values: {e}")
    return None

# Get the proximity sensors' values from robotino
def get_proximity_sensor_values():
    try:
        # Send HTTP GET request to retrieve proximity sensor values
        url = f"http://{IP_ADDRESS}/data/distancesensorarray"
        response = requests.get(url)
        # Check if the response is successful
        if response.status_code == 200:
            # Assuming the response returns an array of floats in a JSON format
            sensor_values = response.json()  # Parse JSON response
            if len(sensor_values) == 9:
                return sensor_values
            else:
                print("Unexpected number of sensor values received.")
        else:
            print(f"Error: Received status code {response.status_code}")
    except Exception as e:
        print(f"Error retrieving sensor values: {e}")
    return None

# Sending commands to Robotino
def send_velocity(vx, vy, omega):
    url = f"http://{IP_ADDRESS}/data/omnidrive"
    data = [vx, vy, omega]  # Prepare the data as a list
    try:
        # Send the velocity data to Robotino
        response = requests.post(url, json=data)  # Send data as JSON
        if response.status_code == 200:
            print(f"Sent Vx: {vx}, Vy: {vy}")
        else:
            print(f"Failed to send data: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending data: {e}")