#!/usr/bin/env python3
"""Start the Flask server in a non-blocking way."""
import subprocess
import sys
import time
import os

# Write a server script
server_code = '''
import sys
sys.path.insert(0, ".")
from src.web.app import create_app
app = create_app()
app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
'''

with open("server.py", "w") as f:
    f.write(server_code)

# Start server as a separate process
proc = subprocess.Popen(
    [sys.executable, "server.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
)

print(f"Server started with PID: {proc.pid}")
print("Waiting for server to be ready...")

# Wait for server to start
time.sleep(3)

# Check if server is running
try:
    import urllib.request
    resp = urllib.request.urlopen("http://127.0.0.1:5001/")
    print(f"Server is running! Status: {resp.status}")
    print(f"Open: http://127.0.0.1:5001")
except Exception as e:
    print(f"Server check failed: {e}")
    print("Server may still be starting...")
