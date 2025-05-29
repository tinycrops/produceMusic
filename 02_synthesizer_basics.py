"""
Synthesizer Fundamentals - Building the Core Components

This module demonstrates essential synthesizer concepts:
- ADSR Envelopes (Attack, Decay, Sustain, Release)
- Low Frequency Oscillators (LFO)
- Frequency Modulation (FM)
- Basic filtering concepts

These are the building blocks of classic synthwave sounds.
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
        """Generate a pure sine wave"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        return t, wave
    
    def generate_square_wave(self, frequency, duration, amplitude=0.5):
        """Generate a square wave"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        sine = np.sin(2 * np.pi * frequency * t)
        square = amplitude * np.sign(sine)
        return t, square
    
    def generate_sawtooth_wave(self, frequency, duration, amplitude=0.5):
        """Generate a sawtooth wave"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        sawtooth = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
        return t, sawtooth
    
    def play_wave(self, wave_data):
        """Play audio through speakers"""
        sd.play(wave_data, self.sample_rate)
        sd.wait()
    
    def save_wave(self, wave_data, filename):
        """Save audio to a WAV file"""
        sf.write(filename, wave_data, self.sample_rate)
        print(f"Saved audio to {filename}")

class Synthesizer(DigitalAudio):
    def __init__(self, sample_rate=44100):
        super().__init__(sample_rate)
    
    def adsr_envelope(self, duration, attack=0.1, decay=0.2, sustain_level=0.7, release=0.5):
        """
        ADSR Envelope - Controls how a note's volume changes over time
        This is what makes the difference between a piano hit and a string pad
        
        Args:
            duration: Total note duration in seconds
            attack: Time to reach peak volume (seconds)
            decay: Time to drop to sustain level (seconds)  
            sustain_level: Volume level to hold (0.0 to 1.0)
            release: Time to fade to silence (seconds)
        """
        total_samples = int(duration * self.sample_rate)
        envelope = np.zeros(total_samples)
        
        # Calculate sample indices for each phase
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate) 
        release_samples = int(release * self.sample_rate)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        current_sample = 0
        
        # Attack phase: 0 → 1
        if attack_samples > 0:
            envelope[current_sample:current_sample + attack_samples] = np.linspace(0, 1, attack_samples)
            current_sample += attack_samples
        
        # Decay phase: 1 → sustain_level
        if decay_samples > 0:
            envelope[current_sample:current_sample + decay_samples] = np.linspace(1, sustain_level, decay_samples)
            current_sample += decay_samples
        
        # Sustain phase: hold at sustain_level
        if sustain_samples > 0:
            envelope[current_sample:current_sample + sustain_samples] = sustain_level
            current_sample += sustain_samples
        
        # Release phase: sustain_level → 0
        if release_samples > 0 and current_sample < total_samples:
            remaining = total_samples - current_sample
            envelope[current_sample:current_sample + remaining] = np.linspace(sustain_level, 0, remaining)
        
        return envelope
    
    def low_frequency_oscillator(self, frequency, duration, waveform='sine'):
        """
        LFO - Low Frequency Oscillator (usually below 20 Hz)
        Used to modulate other parameters like pitch, volume, or filter cutoff
        Essential for creating vibrato, tremolo, and animated synthwave sounds
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        if waveform == 'sine':
            return np.sin(2 * np.pi * frequency * t)
        elif waveform == 'triangle':
            return 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
        elif waveform == 'square':
            return np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform == 'sawtooth':
            return 2 * (t * frequency - np.floor(t * frequency + 0.5))
    
    def frequency_modulation(self, carrier_freq, modulator_freq, mod_depth, duration):
        """
        Frequency Modulation (FM) - Changes the pitch of one oscillator with another
        This creates complex, harmonic-rich timbres popular in 80s synths
        
        Args:
            carrier_freq: The main frequency we hear
            modulator_freq: The frequency doing the modulation
            mod_depth: How much the pitch changes (in Hz)
            duration: Length in seconds
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create modulator signal
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        
        # Modulate the carrier frequency
        instantaneous_freq = carrier_freq + (mod_depth * modulator)
        
        # Generate the FM sound by integrating the instantaneous frequency
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
        fm_wave = np.sin(phase)
        
        return t, fm_wave
    
    def simple_lowpass_filter(self, audio, cutoff_freq, resonance=1.0):
        """
        Simple low-pass filter - removes high frequencies
        Higher resonance emphasizes the cutoff frequency
        Essential for warm, analog-sounding synthwave tones
        """
        # This is a simplified version - real filters use more complex algorithms
        from scipy import signal
        
        # Normalize cutoff frequency (Nyquist frequency = sample_rate/2)
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Create butterworth filter
        b, a = signal.butter(2, normalized_cutoff, btype='low')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio)
        
        # Apply resonance (simplified)
        if resonance > 1.0:
            # Boost frequencies around cutoff
            b_peak, a_peak = signal.butter(2, normalized_cutoff, btype='bandpass')
            peak = signal.filtfilt(b_peak, a_peak, audio)
            filtered = filtered + (resonance - 1.0) * 0.3 * peak
        
        return filtered
    
    def create_synthwave_bass(self, frequency, duration, cutoff=800, resonance=2.0):
        """
        Create a classic synthwave bass sound:
        - Sawtooth wave (bright and rich)
        - Low-pass filter with resonance
        - Punchy ADSR envelope
        """
        # Generate sawtooth wave
        t, sawtooth = self.generate_sawtooth_wave(frequency, duration, amplitude=0.8)
        
        # Apply low-pass filter for warmth
        filtered = self.simple_lowpass_filter(sawtooth, cutoff, resonance)
        
        # Apply punchy envelope (quick attack, medium decay, no sustain, quick release)
        envelope = self.adsr_envelope(duration, attack=0.01, decay=0.3, sustain_level=0.0, release=0.1)
        
        return filtered * envelope
    
    def create_synthwave_lead(self, frequency, duration):
        """
        Create a classic synthwave lead sound:
        - Square wave with some FM modulation
        - Moderate filter
        - Sustained envelope with vibrato
        """
        # Generate square wave
        t, square = self.generate_square_wave(frequency, duration, amplitude=0.6)
        
        # Add slight FM modulation for character
        t_fm, fm_component = self.frequency_modulation(frequency, frequency * 0.5, 10, duration)
        combined = square + 0.2 * fm_component
        
        # Apply filter
        filtered = self.simple_lowpass_filter(combined, 2000, 1.5)
        
        # Apply sustained envelope
        envelope = self.adsr_envelope(duration, attack=0.1, decay=0.2, sustain_level=0.8, release=0.3)
        
        # Add vibrato (pitch modulation)
        vibrato = self.low_frequency_oscillator(5, duration, 'sine')  # 5 Hz vibrato
        # Approximate vibrato by very slight volume modulation
        vibrato_envelope = 1.0 + 0.05 * vibrato
        
        return filtered * envelope * vibrato_envelope

def main():
    """Demonstrate synthesizer concepts crucial for synthwave"""
    print("=== Synthesizer Fundamentals for Synthwave ===")
    
    synth = Synthesizer()
    
    # Demonstrate ADSR envelope
    print("1. ADSR Envelope - Controls how notes evolve over time")
    duration = 3.0
    envelope = synth.adsr_envelope(duration, attack=0.2, decay=0.5, sustain_level=0.6, release=1.0)
    
    # Visualize the envelope
    t = np.linspace(0, duration, len(envelope))
    plt.figure(figsize=(10, 4))
    plt.plot(t, envelope)
    plt.title("ADSR Envelope")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    
    # Demonstrate basic FM synthesis
    print("2. Frequency Modulation - Creates complex harmonic content")
    t_fm, fm_sound = synth.frequency_modulation(440, 220, 50, 2.0)
    synth.play_wave(fm_sound * 0.5)  # Lower volume for comfort
    
    # Create synthwave bass
    print("3. Synthwave Bass - Filtered sawtooth with punchy envelope")
    bass = synth.create_synthwave_bass(80, 2.0)  # Low E
    synth.play_wave(bass)
    synth.save_wave(bass, "synthwave_bass.wav")
    
    # Create synthwave lead
    print("4. Synthwave Lead - Square wave with character")
    lead = synth.create_synthwave_lead(440, 3.0)  # A4
    synth.play_wave(lead)
    synth.save_wave(lead, "synthwave_lead.wav")
    
    print("\n=== Key Synthwave Elements Demonstrated ===")
    print("• ADSR Envelopes: Shape how notes evolve (punchy bass vs sustained pads)")
    print("• Filtering: Removes harsh frequencies, adds warmth")
    print("• FM Synthesis: Creates complex, evolving timbres")
    print("• LFO Modulation: Adds movement and life to static sounds")
    print("• Waveform Selection: Sawtooth for bass, square for leads")

if __name__ == "__main__":
    main() 