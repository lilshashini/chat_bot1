import time
import subprocess
import psutil
from pyngrok import ngrok, conf

# Step 1: Set your ngrok auth token (get it from https://dashboard.ngrok.com/get-started/your-authtoken)

NGROK_AUTH_TOKEN = "2yrmyGhanFHy9W3XNUsLgJFYIjk_2mzApqLKXtkrWYuQdidEA"  # 🔐 Replace with your token!
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# Fix for macOS: Proper port killing function
def kill_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    print(f"Killing process {proc.pid} on port {port}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

kill_port(8501)


# Step 3: Start Streamlit app
print("🚀 Launching Streamlit app...")
streamlit_process = subprocess.Popen(["streamlit", "run", "ai_app.py"])

# Step 4: Wait a bit to make sure Streamlit starts
time.sleep(5)

# Step 5: Start ngrok tunnel
public_url = ngrok.connect(8501, bind_tls=True)
print(f"\n🌍 Your Streamlit app is live at: {public_url}")

# Step 6: Keep the script running to keep tunnel alive
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("🛑 Stopping app...")
    streamlit_process.terminate()
    ngrok.kill()
