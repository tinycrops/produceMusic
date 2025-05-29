"""
Real-time Audio Control - Live Synthwave Performance

This module demonstrates:
- Real-time audio generation and processing
- Parameter control while audio is playing
- Building a simple live performance interface
- Understanding audio buffers and latency

This shows how to create responsive, interactive audio tools.
"""

import numpy as np
import sounddevice as sd
import threading
import time
import queue
import matplotlib.pyplot as plt

class RealtimeSynth:
    """Real-time synthesizer with controllable parameters"""
    
    def __init__(self, sample_rate=44100, block_size=1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Synthesizer state
        self.frequency = 440.0
        self.amplitude = 0.3
        self.waveform = 'sawtooth'  # 'sine', 'square', 'sawtooth', 'triangle'
        self.filter_cutoff = 2000.0
        self.filter_resonance = 1.0
        
        # Real-time control
        self.is_playing = False
        self.phase = 0.0
        
        # Parameter smoothing to avoid clicks
        self.smooth_freq = self.frequency
        self.smooth_amp = self.amplitude
        self.smooth_cutoff = self.filter_cutoff
        
    def generate_waveform(self, freq, num_samples):
        """Generate one block of audio"""
        # Calculate time array for this block
        t = np.arange(num_samples) / self.sample_rate
        
        # Update phase to maintain continuity
        phase_increment = 2 * np.pi * freq / self.sample_rate
        phases = self.phase + np.arange(num_samples) * phase_increment
        self.phase = phases[-1] % (2 * np.pi)
        
        # Generate waveform
        if self.waveform == 'sine':
            wave = np.sin(phases)
        elif self.waveform == 'square':
            wave = np.sign(np.sin(phases))
        elif self.waveform == 'sawtooth':
            wave = 2 * (phases / (2 * np.pi) - np.floor(phases / (2 * np.pi) + 0.5))
        elif self.waveform == 'triangle':
            saw = 2 * (phases / (2 * np.pi) - np.floor(phases / (2 * np.pi) + 0.5))
            wave = 2 * np.abs(saw) - 1
        else:
            wave = np.sin(phases)
        
        return wave
    
    def simple_lowpass_filter(self, audio, cutoff):
        """Simple one-pole low-pass filter for real-time use"""
        # Calculate filter coefficient
        omega = 2 * np.pi * cutoff / self.sample_rate
        alpha = 1 - np.exp(-omega)
        
        # Apply filter (simplified)
        filtered = np.zeros_like(audio)
        if hasattr(self, 'filter_state'):
            filtered[0] = alpha * audio[0] + (1 - alpha) * self.filter_state
        else:
            filtered[0] = audio[0]
            
        for i in range(1, len(audio)):
            filtered[i] = alpha * audio[i] + (1 - alpha) * filtered[i-1]
        
        # Store last sample for next block
        self.filter_state = filtered[-1]
        
        return filtered
    
    def audio_callback(self, outdata, frames, time, status):
        """Called by sounddevice for each audio block"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_playing:
            # Smooth parameter changes to avoid clicks
            smoothing = 0.99
            self.smooth_freq = self.smooth_freq * smoothing + self.frequency * (1 - smoothing)
            self.smooth_amp = self.smooth_amp * smoothing + self.amplitude * (1 - smoothing)
            self.smooth_cutoff = self.smooth_cutoff * smoothing + self.filter_cutoff * (1 - smoothing)
            
            # Generate audio
            wave = self.generate_waveform(self.smooth_freq, frames)
            
            # Apply filter
            if self.smooth_cutoff < self.sample_rate / 2:
                wave = self.simple_lowpass_filter(wave, self.smooth_cutoff)
            
            # Apply amplitude
            wave *= self.smooth_amp
            
            # Output (mono to stereo)
            outdata[:, 0] = wave
            outdata[:, 1] = wave
        else:
            # Silence
            outdata.fill(0)
    
    def start(self):
        """Start real-time audio"""
        self.stream = sd.OutputStream(
            callback=self.audio_callback,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=2,
            dtype=np.float32
        )
        self.stream.start()
        self.is_playing = True
        print("Real-time audio started")
    
    def stop(self):
        """Stop real-time audio"""
        self.is_playing = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("Real-time audio stopped")
    
    def set_frequency(self, freq):
        """Change frequency in real-time"""
        self.frequency = max(20, min(freq, 4000))  # Clamp to reasonable range
    
    def set_amplitude(self, amp):
        """Change amplitude in real-time"""
        self.amplitude = max(0, min(amp, 1.0))
    
    def set_filter_cutoff(self, cutoff):
        """Change filter cutoff in real-time"""
        self.filter_cutoff = max(100, min(cutoff, self.sample_rate / 2))
    
    def set_waveform(self, waveform):
        """Change waveform in real-time"""
        if waveform in ['sine', 'square', 'sawtooth', 'triangle']:
            self.waveform = waveform

class SynthwaveSequencer:
    """Real-time step sequencer"""
    
    def __init__(self, synth, bpm=120):
        self.synth = synth
        self.bpm = bpm
        self.step_duration = 60.0 / bpm / 4  # 16th note
        
        # Sequencer state
        self.pattern = [440, 0, 330, 0, 294, 0, 392, 0]  # Frequencies (0 = rest)
        self.current_step = 0
        self.is_running = False
        
        # Threading
        self.seq_thread = None
        
    def sequencer_loop(self):
        """Main sequencer loop"""
        while self.is_running:
            # Get current step
            freq = self.pattern[self.current_step]
            
            if freq > 0:
                self.synth.set_frequency(freq)
                self.synth.set_amplitude(0.5)
            else:
                self.synth.set_amplitude(0.0)
            
            # Advance step
            self.current_step = (self.current_step + 1) % len(self.pattern)
            
            # Wait for next step
            time.sleep(self.step_duration)
    
    def start(self):
        """Start sequencer"""
        if not self.is_running:
            self.is_running = True
            self.seq_thread = threading.Thread(target=self.sequencer_loop)
            self.seq_thread.daemon = True
            self.seq_thread.start()
            print("Sequencer started")
    
    def stop(self):
        """Stop sequencer"""
        self.is_running = False
        if self.seq_thread:
            self.seq_thread.join(timeout=1.0)
        print("Sequencer stopped")
    
    def set_pattern(self, pattern):
        """Change the pattern in real-time"""
        self.pattern = pattern
        print(f"Pattern updated: {pattern}")

def interactive_demo():
    """Interactive demo with keyboard controls"""
    print("=== Real-time Synthwave Control Demo ===")
    print("Starting real-time synthesizer...")
    
    # Create synthesizer
    synth = RealtimeSynth()
    synth.start()
    
    print("\nKeyboard controls:")
    print("1-4: Change waveform (1=sine, 2=square, 3=sawtooth, 4=triangle)")
    print("q/a: Frequency up/down")
    print("w/s: Volume up/down") 
    print("e/d: Filter cutoff up/down")
    print("r: Start/stop sequencer")
    print("x: Exit")
    print("\nPress keys and hear the changes in real-time!")
    
    # Create sequencer
    sequencer = SynthwaveSequencer(synth)
    
    try:
        # Simple keyboard input loop
        # Note: This is a basic implementation - a real GUI would be better
        while True:
            command = input("Command: ").strip().lower()
            
            if command == '1':
                synth.set_waveform('sine')
                print("Waveform: Sine")
            elif command == '2':
                synth.set_waveform('square')
                print("Waveform: Square")
            elif command == '3':
                synth.set_waveform('sawtooth')
                print("Waveform: Sawtooth")
            elif command == '4':
                synth.set_waveform('triangle')
                print("Waveform: Triangle")
            elif command == 'q':
                synth.set_frequency(synth.frequency * 1.1)
                print(f"Frequency: {synth.frequency:.1f} Hz")
            elif command == 'a':
                synth.set_frequency(synth.frequency * 0.9)
                print(f"Frequency: {synth.frequency:.1f} Hz")
            elif command == 'w':
                synth.set_amplitude(min(1.0, synth.amplitude + 0.1))
                print(f"Volume: {synth.amplitude:.2f}")
            elif command == 's':
                synth.set_amplitude(max(0.0, synth.amplitude - 0.1))
                print(f"Volume: {synth.amplitude:.2f}")
            elif command == 'e':
                synth.set_filter_cutoff(synth.filter_cutoff * 1.2)
                print(f"Filter cutoff: {synth.filter_cutoff:.1f} Hz")
            elif command == 'd':
                synth.set_filter_cutoff(synth.filter_cutoff * 0.8)
                print(f"Filter cutoff: {synth.filter_cutoff:.1f} Hz")
            elif command == 'r':
                if sequencer.is_running:
                    sequencer.stop()
                else:
                    sequencer.start()
            elif command == 'x':
                break
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        sequencer.stop()
        synth.stop()

def automated_demo():
    """Automated demo showing parameter changes over time"""
    print("=== Automated Real-time Demo ===")
    print("This demo will automatically change parameters over time")
    
    synth = RealtimeSynth()
    synth.start()
    
    # Demo script
    demo_steps = [
        (0, lambda: synth.set_waveform('sawtooth')),
        (0, lambda: print("Starting with sawtooth wave...")),
        (2, lambda: synth.set_frequency(220)),
        (2, lambda: print("Low frequency...")),
        (4, lambda: synth.set_frequency(440)),
        (4, lambda: print("Higher frequency...")),
        (6, lambda: synth.set_filter_cutoff(800)),
        (6, lambda: print("Lowering filter cutoff...")),
        (8, lambda: synth.set_filter_cutoff(2000)),
        (8, lambda: print("Opening filter...")),
        (10, lambda: synth.set_waveform('square')),
        (10, lambda: print("Switching to square wave...")),
        (12, lambda: synth.set_amplitude(0.6)),
        (14, lambda: synth.set_amplitude(0.2)),
        (16, lambda: synth.set_amplitude(0.0)),
        (16, lambda: print("Fading out...")),
    ]
    
    try:
        start_time = time.time()
        step_index = 0
        
        while step_index < len(demo_steps):
            current_time = time.time() - start_time
            step_time, action = demo_steps[step_index]
            
            if current_time >= step_time:
                action()
                step_index += 1
            
            time.sleep(0.1)  # Check every 100ms
            
    finally:
        synth.stop()
        print("Demo complete!")

def main():
    """Main demo function"""
    print("=== Real-time Audio Control ===")
    print("Choose demo mode:")
    print("1. Interactive control (keyboard input)")
    print("2. Automated parameter demo")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        interactive_demo()
    elif choice == '2':
        automated_demo()
    else:
        print("Invalid choice")
        
    print("\n=== Real-time Concepts Demonstrated ===")
    print("• Audio Callbacks: Processing audio in small blocks")
    print("• Parameter Smoothing: Avoiding clicks when changing settings")
    print("• Threading: Running sequencer independently of audio")
    print("• Low Latency: Responsive real-time control")
    print("• Buffer Management: Understanding audio block sizes")

if __name__ == "__main__":
    main() 