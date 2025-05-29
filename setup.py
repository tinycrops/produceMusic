#!/usr/bin/env python3
"""
Setup script for Digital Audio & Synthwave Production

This script installs dependencies and runs your first digital audio program.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def run_audio_test():
    """Test audio setup"""
    print("\nTesting audio setup...")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        # Test audio playback
        print("Playing test tone...")
        duration = 1  # seconds
        sample_rate = 44100
        frequency = 440  # A4
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        sd.play(tone, sample_rate)
        sd.wait()
        
        print("✓ Audio system working!")
        return True
        
    except Exception as e:
        print(f"✗ Audio test failed: {e}")
        print("You may need to configure your audio drivers or check your speakers/headphones")
        return False

def main():
    print("=== Digital Audio & Synthwave Production Setup ===")
    print("This will install dependencies and test your audio setup.")
    print()
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed. Please check your Python installation and try again.")
        return
    
    # Test audio
    if not run_audio_test():
        print("Audio test failed, but you can still continue with visual examples.")
    
    print("\n=== Setup Complete! ===")
    print("Ready to explore digital audio fundamentals!")
    print()
    print("Try running these programs in order:")
    print("1. python 01_digital_audio_basics.py")
    print("2. python 02_synthesizer_basics.py")
    print("3. python 03_effects_and_processing.py")
    print()
    print("Each program will:")
    print("• Demonstrate key concepts with visual plots")
    print("• Play audio examples")
    print("• Save WAV files you can analyze")
    print("• Show you the underlying mathematics and code")

if __name__ == "__main__":
    main() 