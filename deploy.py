#!/usr/bin/env python3
"""
One-click deploy for hackathon presentation.
Creates a public URL accessible from any computer.

Usage:
    python deploy.py

First time: Sign up at https://ngrok.com (free) and get your auth token.
Then run:
    ngrok config add-authtoken YOUR_TOKEN
"""

import subprocess
import sys
import time
import webbrowser

def main():
    print("=" * 50)
    print("  OIL SPILL DETECTION - PUBLIC DEPLOY")
    print("=" * 50)
    print()

    # Start Flask server
    print("Starting server on port 5000...")
    server = subprocess.Popen(
        [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from src.web.app import create_app
app = create_app()
app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
"""],
        cwd=".",
    )
    time.sleep(2)

    # Start ngrok tunnel
    print("Creating public URL with ngrok...")
    try:
        from pyngrok import ngrok
        tunnel = ngrok.connect(5000)
        public_url = tunnel.public_url
        print()
        print("=" * 50)
        print(f"  YOUR PUBLIC URL:")
        print(f"  {public_url}")
        print("=" * 50)
        print()
        print("Share this link with anyone - they can open it on any computer!")
        print("Press Ctrl+C to stop.")
        print()

        # Open in browser
        webbrowser.open(public_url)

        # Keep running
        server.wait()

    except ImportError:
        print("pyngrok not installed. Install with: pip install pyngrok")
        server.terminate()
    except Exception as e:
        print(f"ngrok error: {e}")
        print()
        print("If you see an auth error, run:")
        print("  1. Go to https://ngrok.com (free signup)")
        print("  2. Get your auth token")
        print("  3. Run: ngrok config add-authtoken YOUR_TOKEN")
        print("  4. Run this script again")
        server.terminate()

if __name__ == "__main__":
    main()
