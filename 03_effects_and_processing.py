"""
Audio Effects and Processing - The Magic of Synthwave

This module implements classic audio effects that define the synthwave sound:
- Reverb (especially gated reverb)
- Analog-style delay
- Chorus for width and movement
- Soft distortion/saturation
- Compression for punch

Understanding these effects from first principles gives you complete control
over your sound design.
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy import signal

class AudioEffects:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def delay(self, audio, delay_time, feedback=0.3, mix=0.3):
        """
        Digital delay effect - creates echoes
        
        Args:
            audio: Input audio signal
            delay_time: Delay time in seconds
            feedback: How much delayed signal feeds back (0.0-0.95)
            mix: Wet/dry mix (0.0=dry, 1.0=fully wet)
        """
        delay_samples = int(delay_time * self.sample_rate)
        output = np.copy(audio)
        
        # Create delay buffer
        delay_buffer = np.zeros(len(audio) + delay_samples)
        delay_buffer[:len(audio)] = audio
        
        # Apply feedback delay
        for i in range(delay_samples, len(delay_buffer)):
            if i < len(audio):
                # Add delayed signal with feedback
                delay_buffer[i] += delay_buffer[i - delay_samples] * feedback
                output[i] = audio[i] * (1 - mix) + delay_buffer[i] * mix
        
        return output
    
    def reverb(self, audio, room_size=0.5, damping=0.5, mix=0.3):
        """
        Simple reverb using multiple delay lines
        This simulates sound bouncing around a room
        
        Args:
            audio: Input audio signal
            room_size: Size of the simulated room (0.0-1.0)
            damping: High frequency absorption (0.0-1.0)
            mix: Wet/dry mix
        """
        # Multiple delay times for natural reverb
        delay_times = np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.11]) * (1 + room_size)
        
        reverb_signal = np.zeros_like(audio)
        
        for delay_time in delay_times:
            # Each delay line with different characteristics
            delayed = self.delay(audio, delay_time, feedback=0.2, mix=1.0)
            
            # Apply damping (low-pass filter)
            if damping > 0:
                cutoff = 8000 * (1 - damping)  # Higher damping = lower cutoff
                nyquist = self.sample_rate / 2
                if cutoff < nyquist:
                    b, a = signal.butter(1, cutoff / nyquist, btype='low')
                    delayed = signal.filtfilt(b, a, delayed)
            
            reverb_signal += delayed * 0.2  # Mix multiple delay lines
        
        # Combine dry and wet signals
        return audio * (1 - mix) + reverb_signal * mix
    
    def gated_reverb(self, audio, gate_threshold=0.1, room_size=0.7, mix=0.4):
        """
        Gated reverb - iconic 80s effect
        Reverb is cut off when the signal drops below threshold
        Phil Collins' "In the Air Tonight" made this famous
        """
        # Generate reverb
        reverbed = self.reverb(audio, room_size=room_size, mix=1.0)
        
        # Create gate envelope based on original signal
        envelope = np.abs(audio)
        
        # Smooth the envelope
        window_size = int(0.01 * self.sample_rate)  # 10ms smoothing
        envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Apply gate
        gate = envelope > gate_threshold
        gate = gate.astype(float)
        
        # Smooth gate transitions to avoid clicks
        gate = np.convolve(gate, np.ones(window_size)/window_size, mode='same')
        
        # Apply gate to reverb
        gated_reverb = reverbed * gate
        
        # Mix with dry signal
        return audio * (1 - mix) + gated_reverb * mix
    
    def chorus(self, audio, rate=0.5, depth=0.002, mix=0.5):
        """
        Chorus effect - creates width and movement
        Uses modulated delay lines to simulate multiple voices
        
        Args:
            audio: Input audio signal
            rate: LFO rate in Hz
            depth: Modulation depth in seconds
            mix: Wet/dry mix
        """
        # Generate LFO for modulation
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio), False)
        lfo = np.sin(2 * np.pi * rate * t)
        
        # Convert depth to samples
        depth_samples = depth * self.sample_rate
        
        # Create modulated delay
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Calculate delay with LFO modulation
            delay_samples = int(depth_samples * (1 + lfo[i]))
            
            if i >= delay_samples:
                output[i] = audio[i - delay_samples]
        
        # Mix dry and wet
        return audio * (1 - mix) + output * mix
    
    def soft_distortion(self, audio, drive=2.0, mix=1.0):
        """
        Soft distortion/saturation - adds warmth and character
        Uses hyperbolic tangent for smooth saturation
        
        Args:
            audio: Input audio signal
            drive: Amount of distortion (1.0 = clean, higher = more drive)
            mix: Wet/dry mix
        """
        # Apply drive
        driven = audio * drive
        
        # Soft clipping using tanh
        distorted = np.tanh(driven) / drive  # Normalize by drive amount
        
        # Mix with original
        return audio * (1 - mix) + distorted * mix
    
    def compressor(self, audio, threshold=0.7, ratio=4.0, attack=0.003, release=0.1):
        """
        Dynamic range compressor - evens out volume levels
        Essential for punchy synthwave sounds
        
        Args:
            audio: Input audio signal
            threshold: Level above which compression starts (0.0-1.0)
            ratio: Compression ratio (2.0 = 2:1, 4.0 = 4:1, etc.)
            attack: Time to start compressing (seconds)
            release: Time to stop compressing (seconds)
        """
        # Convert times to samples
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # Calculate envelope of signal
        envelope = np.abs(audio)
        
        # Smooth envelope (peak detector)
        smoothed_env = np.zeros_like(envelope)
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed_env[i-1]:
                # Attack (rising)
                alpha = 1 - np.exp(-1 / attack_samples)
            else:
                # Release (falling)
                alpha = 1 - np.exp(-1 / release_samples)
            
            smoothed_env[i] = alpha * envelope[i] + (1 - alpha) * smoothed_env[i-1]
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(smoothed_env)
        over_threshold = smoothed_env > threshold
        
        # Apply compression to signals over threshold
        gain_reduction[over_threshold] = threshold + (smoothed_env[over_threshold] - threshold) / ratio
        gain_reduction[over_threshold] /= smoothed_env[over_threshold]
        
        return audio * gain_reduction
    
    def stereo_width(self, left, right, width=1.5):
        """
        Stereo width effect - makes mono signals sound wider
        
        Args:
            left, right: Left and right channel audio
            width: Width amount (1.0 = normal, >1.0 = wider, <1.0 = narrower)
        """
        # Calculate mid and side signals
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        
        # Adjust side signal
        side *= width
        
        # Convert back to left/right
        new_left = mid + side
        new_right = mid - side
        
        return new_left, new_right

def create_synthwave_drum_hit():
    """Create a classic synthwave drum sound using effects"""
    sample_rate = 44100
    effects = AudioEffects(sample_rate)
    
    # Start with a short burst of noise (snare-like)
    duration = 0.2
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Mix of sine wave (for punch) and noise (for texture)
    sine_hit = 0.8 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 20)  # Decaying sine
    noise = 0.3 * np.random.normal(0, 0.1, len(t)) * np.exp(-t * 15)  # Decaying noise
    
    drum_hit = sine_hit + noise
    
    # Apply effects
    drum_hit = effects.soft_distortion(drum_hit, drive=3.0)
    drum_hit = effects.compressor(drum_hit, threshold=0.6, ratio=6.0)
    drum_hit = effects.gated_reverb(drum_hit, gate_threshold=0.05, mix=0.6)
    
    return drum_hit

def main():
    """Demonstrate audio effects crucial for synthwave"""
    print("=== Audio Effects for Synthwave ===")
    
    effects = AudioEffects()
    
    # Create a simple test tone
    t = np.linspace(0, 2, int(effects.sample_rate * 2), False)
    test_tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 for 2 seconds
    
    print("1. Original tone")
    sd.play(test_tone, effects.sample_rate)
    sd.wait()
    
    print("2. With analog delay")
    delayed = effects.delay(test_tone, delay_time=0.25, feedback=0.4, mix=0.4)
    sd.play(delayed, effects.sample_rate)
    sd.wait()
    
    print("3. With chorus (adds width and movement)")
    chorused = effects.chorus(test_tone, rate=0.8, depth=0.003, mix=0.6)
    sd.play(chorused, effects.sample_rate)
    sd.wait()
    
    print("4. With soft distortion")
    distorted = effects.soft_distortion(test_tone, drive=4.0)
    sd.play(distorted, effects.sample_rate)
    sd.wait()
    
    print("5. With gated reverb (iconic 80s sound)")
    gated = effects.gated_reverb(test_tone, mix=0.7)
    sd.play(gated, effects.sample_rate)
    sd.wait()
    
    print("6. Synthwave drum hit (combining multiple effects)")
    drum = create_synthwave_drum_hit()
    sd.play(drum, effects.sample_rate)
    sd.wait()
    
    # Save examples
    sf.write("delayed_tone.wav", delayed, effects.sample_rate)
    sf.write("gated_reverb_tone.wav", gated, effects.sample_rate)
    sf.write("synthwave_drum.wav", drum, effects.sample_rate)
    
    print("\n=== Effects Chain Understanding ===")
    print("• Delay: Creates echoes and rhythmic patterns")
    print("• Reverb: Adds space and atmosphere")
    print("• Gated Reverb: Cuts reverb tail for dramatic effect")
    print("• Chorus: Adds width and movement")
    print("• Soft Distortion: Adds warmth and character")
    print("• Compression: Evens out dynamics, adds punch")
    print("• Effects Order Matters: Distortion → Filter → Modulation → Time-based")

if __name__ == "__main__":
    main() 