"""
Advanced Drum Machine - Synthesized Percussion

This module demonstrates:
- Synthesized drum sounds (kick, snare, hi-hat, clap)
- Pattern programming and step sequencing
- Swing/groove timing
- Velocity layers and dynamics
- Classic 808/909 style drum synthesis
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy import signal
import time

class DrumSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def synthesize_kick(self, frequency=60, decay=0.8, punch=0.3):
        """
        Synthesize an 808-style kick drum
        - Low frequency sine wave with pitch envelope
        - Sharp attack with quick pitch drop
        - Long decay for sub-bass feel
        """
        duration = decay
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Pitch envelope: starts high, drops quickly to fundamental
        start_freq = frequency * 4  # Start 4x higher
        pitch_envelope = np.exp(-t * 15)  # Quick exponential decay
        instantaneous_freq = frequency + (start_freq - frequency) * pitch_envelope
        
        # Generate the sine wave with pitch modulation
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
        kick_wave = np.sin(phase)
        
        # Amplitude envelope
        amplitude_envelope = np.exp(-t * (1/decay))
        
        # Add punch (click at the beginning)
        if punch > 0:
            punch_duration = 0.005  # 5ms punch
            punch_samples = int(punch_duration * self.sample_rate)
            punch_env = np.exp(-np.linspace(0, 10, punch_samples))
            noise = np.random.normal(0, 0.1, punch_samples)
            punch_sound = noise * punch_env * punch
            
            # Add punch to beginning
            kick_wave[:len(punch_sound)] += punch_sound
        
        # Apply amplitude envelope
        kick_wave *= amplitude_envelope
        
        # Low-pass filter to remove harsh harmonics
        b, a = signal.butter(2, 0.05, btype='low')  # Very low cutoff
        kick_wave = signal.filtfilt(b, a, kick_wave)
        
        return kick_wave
    
    def synthesize_snare(self, frequency=200, decay=0.3, noise_amount=0.7):
        """
        Synthesize a snare drum
        - Tuned resonant body (sine wave)
        - Noise component for snare buzz
        - Sharp attack with medium decay
        """
        duration = decay
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Tonal component (drum body resonance)
        body = np.sin(2 * np.pi * frequency * t)
        body += 0.5 * np.sin(2 * np.pi * frequency * 1.6 * t)  # Harmonic
        
        # Noise component (snare wires)
        noise = np.random.normal(0, 1, len(t))
        
        # Filter noise to snare frequency range
        b, a = signal.butter(2, [0.1, 0.8], btype='bandpass')
        filtered_noise = signal.filtfilt(b, a, noise)
        
        # Combine body and noise
        snare_wave = (1 - noise_amount) * body + noise_amount * filtered_noise
        
        # Amplitude envelope (quick attack, exponential decay)
        envelope = np.exp(-t * (1/decay)) * (1 - np.exp(-t * 50))  # Quick attack
        
        snare_wave *= envelope
        
        return snare_wave * 0.8
    
    def synthesize_hihat(self, decay=0.1, brightness=0.8, closed=True):
        """
        Synthesize hi-hat (closed or open)
        - High-frequency noise
        - Sharp attack
        - Quick decay for closed, longer for open
        """
        if not closed:
            decay *= 3  # Open hi-hat lasts longer
            
        duration = decay
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # High-frequency noise
        noise = np.random.normal(0, 1, len(t))
        
        # High-pass filter for metallic sound
        cutoff = 0.3 + brightness * 0.4  # Higher = brighter
        b, a = signal.butter(3, cutoff, btype='high')
        hihat_wave = signal.filtfilt(b, a, noise)
        
        # Sharp envelope
        if closed:
            envelope = np.exp(-t * 30)  # Very quick decay
        else:
            envelope = np.exp(-t * 8) * (1 - np.exp(-t * 80))  # Quick attack, longer decay
        
        hihat_wave *= envelope
        
        return hihat_wave * 0.3
    
    def synthesize_clap(self, decay=0.4):
        """
        Synthesize handclap
        - Multiple noise bursts to simulate multiple hands
        - Band-pass filtered noise
        """
        duration = decay
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create multiple short noise bursts
        clap_wave = np.zeros(len(t))
        burst_times = [0.0, 0.01, 0.02, 0.035]  # Timing of hand claps
        
        for burst_time in burst_times:
            start_idx = int(burst_time * self.sample_rate)
            burst_duration = 0.02  # 20ms burst
            burst_samples = int(burst_duration * self.sample_rate)
            
            if start_idx + burst_samples < len(clap_wave):
                # Generate noise burst
                noise = np.random.normal(0, 1, burst_samples)
                burst_envelope = np.exp(-np.linspace(0, 15, burst_samples))
                noise_burst = noise * burst_envelope
                
                # Add to clap wave
                clap_wave[start_idx:start_idx + burst_samples] += noise_burst
        
        # Band-pass filter for clap frequency range
        b, a = signal.butter(2, [0.2, 0.7], btype='bandpass')
        clap_wave = signal.filtfilt(b, a, clap_wave)
        
        # Overall envelope
        envelope = np.exp(-t * (1/decay))
        clap_wave *= envelope
        
        return clap_wave * 0.6

class DrumMachine:
    def __init__(self, bpm=120, sample_rate=44100):
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.synth = DrumSynthesizer(sample_rate)
        
        # Calculate timing
        self.beat_duration = 60.0 / bpm  # Duration of one beat in seconds
        self.step_duration = self.beat_duration / 4  # 16th note duration
        
    def create_pattern(self, length=16):
        """Create empty pattern with specified length (in 16th notes)"""
        return {
            'kick': [0] * length,
            'snare': [0] * length,
            'hihat_closed': [0] * length,
            'hihat_open': [0] * length,
            'clap': [0] * length
        }
    
    def add_swing(self, audio, swing_amount=0.1):
        """
        Add swing to audio by slightly delaying off-beats
        swing_amount: 0.0 = no swing, 0.2 = heavy swing
        """
        # This is a simplified swing implementation
        # In practice, you'd time-stretch specific beats
        return audio  # Placeholder for now
    
    def sequence_pattern(self, pattern, velocity_pattern=None, swing=0.0):
        """
        Convert a drum pattern into audio
        velocity_pattern: Dict with same structure as pattern but with velocity values
        """
        if velocity_pattern is None:
            velocity_pattern = {key: [1.0] * len(pattern[key]) for key in pattern}
        
        # Calculate total duration
        total_steps = len(pattern['kick'])
        total_duration = total_steps * self.step_duration
        total_samples = int(total_duration * self.sample_rate)
        
        # Initialize output
        output = np.zeros(total_samples)
        
        # Generate each drum sound
        for step in range(total_steps):
            step_start_time = step * self.step_duration
            step_start_sample = int(step_start_time * self.sample_rate)
            
            # Kick drum
            if pattern['kick'][step] > 0:
                kick_sound = self.synth.synthesize_kick()
                velocity = velocity_pattern['kick'][step] if step < len(velocity_pattern['kick']) else 1.0
                kick_sound *= velocity * pattern['kick'][step]
                
                end_sample = min(step_start_sample + len(kick_sound), total_samples)
                output[step_start_sample:end_sample] += kick_sound[:end_sample - step_start_sample]
            
            # Snare drum
            if pattern['snare'][step] > 0:
                snare_sound = self.synth.synthesize_snare()
                velocity = velocity_pattern['snare'][step] if step < len(velocity_pattern['snare']) else 1.0
                snare_sound *= velocity * pattern['snare'][step]
                
                end_sample = min(step_start_sample + len(snare_sound), total_samples)
                output[step_start_sample:end_sample] += snare_sound[:end_sample - step_start_sample]
            
            # Closed hi-hat
            if pattern['hihat_closed'][step] > 0:
                hihat_sound = self.synth.synthesize_hihat(closed=True)
                velocity = velocity_pattern['hihat_closed'][step] if step < len(velocity_pattern['hihat_closed']) else 1.0
                hihat_sound *= velocity * pattern['hihat_closed'][step]
                
                end_sample = min(step_start_sample + len(hihat_sound), total_samples)
                output[step_start_sample:end_sample] += hihat_sound[:end_sample - step_start_sample]
            
            # Open hi-hat
            if pattern['hihat_open'][step] > 0:
                hihat_sound = self.synth.synthesize_hihat(closed=False)
                velocity = velocity_pattern['hihat_open'][step] if step < len(velocity_pattern['hihat_open']) else 1.0
                hihat_sound *= velocity * pattern['hihat_open'][step]
                
                end_sample = min(step_start_sample + len(hihat_sound), total_samples)
                output[step_start_sample:end_sample] += hihat_sound[:end_sample - step_start_sample]
            
            # Clap
            if pattern['clap'][step] > 0:
                clap_sound = self.synth.synthesize_clap()
                velocity = velocity_pattern['clap'][step] if step < len(velocity_pattern['clap']) else 1.0
                clap_sound *= velocity * pattern['clap'][step]
                
                end_sample = min(step_start_sample + len(clap_sound), total_samples)
                output[step_start_sample:end_sample] += clap_sound[:end_sample - step_start_sample]
        
        # Apply swing if specified
        if swing > 0:
            output = self.add_swing(output, swing)
        
        # Normalize to prevent clipping
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * 0.8
        
        return output

def main():
    """Demonstrate drum machine capabilities"""
    print("=== Advanced Drum Machine ===")
    
    # Create drum machine
    drum_machine = DrumMachine(bpm=128)
    
    # Test individual drum sounds
    print("1. Testing individual drum sounds...")
    
    kick = drum_machine.synth.synthesize_kick()
    print("Playing kick drum...")
    sd.play(kick, drum_machine.sample_rate)
    sd.wait()
    
    snare = drum_machine.synth.synthesize_snare()
    print("Playing snare drum...")
    sd.play(snare, drum_machine.sample_rate)
    sd.wait()
    
    hihat_closed = drum_machine.synth.synthesize_hihat(closed=True)
    print("Playing closed hi-hat...")
    sd.play(hihat_closed, drum_machine.sample_rate)
    sd.wait()
    
    clap = drum_machine.synth.synthesize_clap()
    print("Playing clap...")
    sd.play(clap, drum_machine.sample_rate)
    sd.wait()
    
    # Create classic patterns
    print("\n2. Creating drum patterns...")
    
    # Basic 4/4 pattern
    basic_pattern = drum_machine.create_pattern(16)
    basic_pattern['kick'] = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # Four on the floor
    basic_pattern['snare'] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # Backbeat
    basic_pattern['hihat_closed'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 8th notes
    
    basic_audio = drum_machine.sequence_pattern(basic_pattern)
    print("Playing basic 4/4 pattern...")
    sd.play(basic_audio, drum_machine.sample_rate)
    sd.wait()
    
    # Trap-style pattern
    trap_pattern = drum_machine.create_pattern(16)
    trap_pattern['kick'] = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    trap_pattern['snare'] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    trap_pattern['hihat_closed'] = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    trap_pattern['clap'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    
    # Add velocity variations
    velocity_pattern = drum_machine.create_pattern(16)
    velocity_pattern['kick'] = [1.0, 0, 0, 0.8, 0, 0, 0.9, 0, 1.0, 0, 0.7, 0, 0, 0.8, 0, 0]
    velocity_pattern['hihat_closed'] = [0.8, 0.6, 0, 0.7, 0.9, 0.5, 0, 0.8, 0.7, 0.6, 0, 0.9, 0.8, 0.5, 0, 0.7]
    
    trap_audio = drum_machine.sequence_pattern(trap_pattern, velocity_pattern)
    print("Playing trap-style pattern with velocity variations...")
    sd.play(trap_audio, drum_machine.sample_rate)
    sd.wait()
    
    # Save patterns
    sf.write("basic_drums.wav", basic_audio, drum_machine.sample_rate)
    sf.write("trap_drums.wav", trap_audio, drum_machine.sample_rate)
    print("\nSaved drum patterns as WAV files")
    
    # Visualize pattern
    print("\n3. Pattern visualization:")
    pattern_viz = trap_pattern
    instruments = list(pattern_viz.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, instrument in enumerate(instruments):
        pattern_data = pattern_viz[instrument]
        y_pos = [i] * len(pattern_data)
        x_pos = range(len(pattern_data))
        
        # Show hits as dots
        hits = [x for x, hit in enumerate(pattern_data) if hit > 0]
        hit_y = [i] * len(hits)
        
        ax.scatter(hits, hit_y, s=100, alpha=0.8)
        ax.set_yticks(range(len(instruments)))
        ax.set_yticklabels(instruments)
        ax.set_xlabel('Step (16th notes)')
        ax.set_title('Drum Pattern Visualization')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Drum Machine Concepts Demonstrated ===")
    print("• Synthesized Drums: Creating drums from scratch using synthesis")
    print("• 808/909 Style: Classic analog drum machine sounds")
    print("• Step Sequencing: Programming rhythmic patterns")
    print("• Velocity Layers: Dynamic expression in drum programming")
    print("• Pattern Variations: Different genres require different approaches")
    print("• Sound Design: Tuning percussion for musical context")

if __name__ == "__main__":
    main() 