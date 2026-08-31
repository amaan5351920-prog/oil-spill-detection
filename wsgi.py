#!/usr/bin/env python3
"""WSGI entry point for production deployment."""
from src.web.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
