"""
Digital Audio Fundamentals - Understanding How Sound Becomes Numbers

This module demonstrates the core concepts of digital audio:
- Sampling rate and bit depth
- Basic waveform generation
- How continuous sound becomes discrete digital data
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

class DigitalAudio:
    def __init__(self, sample_rate=44100):
        """
        Initialize with standard CD quality sample rate
        44.1kHz means we capture 44,100 samples per second
        """
        self.sample_rate = sample_rate
        
    def generate_sine_wave(self, frequency, duration, amplitude=0.5):
        """
        Generate a pure sine wave - the fundamental building block of all sound
        
        Args:
            frequency: Pitch in Hz (A4 = 440 Hz)
            duration: Length in seconds
            amplitude: Volume (0.0 to 1.0)
        """
        # Create time array: duration * sample_rate gives us total samples needed
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Generate sine wave: amplitude * sin(2π * frequency * time)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        return t, wave
    
    def generate_square_wave(self, frequency, duration, amplitude=0.5):
        """
        Square wave - creates that classic digital/chiptune sound
        Mathematically: sign(sin(2πft))
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        sine = np.sin(2 * np.pi * frequency * t)
        square = amplitude * np.sign(sine)
        return t, square
    
    def generate_sawtooth_wave(self, frequency, duration, amplitude=0.5):
        """
        Sawtooth wave - common in analog synthesizers
        Creates a bright, buzzy sound rich in harmonics
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Sawtooth: 2 * (t * frequency - floor(t * frequency + 0.5))
        sawtooth = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
        return t, sawtooth
    
    def generate_triangle_wave(self, frequency, duration, amplitude=0.5):
        """
        Triangle wave - softer than square, contains only odd harmonics
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        sawtooth = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        triangle = amplitude * 2 * np.abs(sawtooth) - amplitude
        return t, triangle
    
    def play_wave(self, wave_data):
        """Play audio through your speakers/headphones"""
        sd.play(wave_data, self.sample_rate)
        sd.wait()  # Wait until the sound finishes
    
    def save_wave(self, wave_data, filename):
        """Save audio to a WAV file"""
        sf.write(filename, wave_data, self.sample_rate)
        print(f"Saved audio to {filename}")
    
    def visualize_waves(self, waves_dict, duration=0.01):
        """
        Visualize different waveforms to understand their shapes
        Only show first 10ms to see the wave shape clearly
        """
        fig, axes = plt.subplots(len(waves_dict), 1, figsize=(12, 8))
        if len(waves_dict) == 1:
            axes = [axes]
        
        for i, (name, (t, wave)) in enumerate(waves_dict.items()):
            # Only show first portion for visualization
            samples_to_show = int(duration * self.sample_rate)
            axes[i].plot(t[:samples_to_show], wave[:samples_to_show])
            axes[i].set_title(f"{name} Wave")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True)
        
        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.show()

def main():
    """Demonstrate basic digital audio concepts"""
    print("=== Digital Audio Fundamentals ===")
    print(f"Sample Rate: 44,100 samples per second")
    print(f"This means we measure the sound wave 44,100 times every second")
    print()
    
    # Create our digital audio engine
    audio = DigitalAudio()
    
    # Generate different waveforms at A4 (440 Hz)
    frequency = 440  # A4 note
    duration = 2.0   # 2 seconds
    
    print("Generating fundamental waveforms...")
    t_sine, sine = audio.generate_sine_wave(frequency, duration)
    t_square, square = audio.generate_square_wave(frequency, duration)
    t_saw, sawtooth = audio.generate_sawtooth_wave(frequency, duration)
    t_tri, triangle = audio.generate_triangle_wave(frequency, duration)
    
    # Visualize the waveforms
    waves = {
        "Sine": (t_sine, sine),
        "Square": (t_square, square), 
        "Sawtooth": (t_saw, sawtooth),
        "Triangle": (t_tri, triangle)
    }
    
    print("Displaying waveform shapes...")
    audio.visualize_waves(waves)
    
    # Play each waveform
    print("\nPlaying waveforms...")
    print("1. Sine wave (pure tone, like a tuning fork)")
    audio.play_wave(sine)
    
    print("2. Square wave (digital, chiptune-like)")
    audio.play_wave(square)
    
    print("3. Sawtooth wave (bright, buzzy)")
    audio.play_wave(sawtooth)
    
    print("4. Triangle wave (softer than square)")
    audio.play_wave(triangle)
    
    # Save examples
    audio.save_wave(sine, "sine_440hz.wav")
    audio.save_wave(square, "square_440hz.wav")
    
    print("\n=== Key Concepts Demonstrated ===")
    print("• Sample Rate: How often we measure the sound wave")
    print("• Waveforms: Different shapes create different timbres")
    print("• Digital Representation: Continuous sound → discrete numbers")
    print("• Frequency: Determines pitch (440 Hz = A4)")
    print("• Amplitude: Determines volume")

if __name__ == "__main__":
    main() 