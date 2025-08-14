#!/usr/bin/env python3
"""
Alpha Sentinel Bot Startup Script
This script provides a simple way to start the bot with different modes
"""

import pandas as pd
import subprocess
import sys
import os
import argparse

_original_fillna = pd.DataFrame.fillna


def start_bot():
    """Start the main bot"""
    print("🤖 Starting Alpha Sentinel Bot...")
    subprocess.run([sys.executable, "src/main.py"])


def start_dashboard():
    """Start the dashboard"""
    print("📊 Starting Alpha Sentinel Dashboard...")
    subprocess.run([sys.executable, "run_dashboard.py"])


def train_models():
    """Train ML models"""
    print("🧠 Training ML Models...")
    subprocess.run([sys.executable, "train_models.py"])


def test_system():
    """Test all system components"""
    print("🧪 Testing System Components...")
    subprocess.run([sys.executable, "test_system.py"])


def main():
    parser = argparse.ArgumentParser(description="Alpha Sentinel Bot Control Script")
    parser.add_argument(
        "command",
        choices=["bot", "dashboard", "train", "test", "all"],
        help="Command to execute",
    )

    args = parser.parse_args()

    if args.command == "bot":
        start_bot()
    elif args.command == "dashboard":
        start_dashboard()
    elif args.command == "train":
        train_models()
    elif args.command == "test":
        test_system()
    elif args.command == "all":
        print("🚀 Starting complete Alpha Sentinel system...")
        print("   Step 1: Testing system...")
        test_system()
        print("   Step 2: Training models...")
        train_models()
        print("   Step 3: Starting bot in background...")
        # Start bot in background and dashboard in foreground
        import threading

        bot_thread = threading.Thread(target=start_bot)
        bot_thread.daemon = True
        bot_thread.start()

        print("   Step 4: Starting dashboard...")
        start_dashboard()


if __name__ == "__main__":
    main()
