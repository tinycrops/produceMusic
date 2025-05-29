"""
Sequencer and Composition - Building Synthwave Tracks

This module demonstrates:
- Step sequencing (the heart of electronic music)
- Musical scales and chord progressions
- Layering multiple synthesizer parts
- Creating a complete synthwave composition

This gives you the tools to compose full tracks programmatically.
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# Import our previous synthesizer classes
import sys
sys.path.append('.')

class MusicalScale:
    """Musical scale and chord utilities"""
    
    # Note frequencies (A4 = 440 Hz)
    NOTE_FREQUENCIES = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    
    # Scale patterns (semitone intervals)
    SCALES = {
        'minor': [0, 2, 3, 5, 7, 8, 10],  # Natural minor
        'major': [0, 2, 4, 5, 7, 9, 11],  # Major
        'dorian': [0, 2, 3, 5, 7, 9, 10],  # Dorian (popular in synthwave)
    }
    
    @classmethod
    def get_frequency(cls, note, octave=4):
        """Get frequency for a note in a specific octave"""
        base_freq = cls.NOTE_FREQUENCIES[note]
        # Each octave doubles the frequency
        return base_freq * (2 ** (octave - 4))
    
    @classmethod
    def get_scale(cls, root_note, scale_type='minor', octave=4):
        """Get frequencies for a scale"""
        root_freq = cls.get_frequency(root_note, octave)
        scale_intervals = cls.SCALES[scale_type]
        
        frequencies = []
        for interval in scale_intervals:
            # Each semitone is a factor of 2^(1/12)
            freq = root_freq * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies
    
    @classmethod
    def get_chord(cls, root_note, chord_type='minor', octave=4):
        """Get frequencies for a chord"""
        root_freq = cls.get_frequency(root_note, octave)
        
        if chord_type == 'minor':
            intervals = [0, 3, 7]  # Root, minor third, fifth
        elif chord_type == 'major':
            intervals = [0, 4, 7]  # Root, major third, fifth
        elif chord_type == 'minor7':
            intervals = [0, 3, 7, 10]  # Add minor seventh
        elif chord_type == 'major7':
            intervals = [0, 4, 7, 11]  # Add major seventh
        
        frequencies = []
        for interval in intervals:
            freq = root_freq * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies

class StepSequencer:
    """8-step sequencer for programming beats and patterns"""
    
    def __init__(self, sample_rate=44100, bpm=120):
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.step_duration = 60.0 / bpm / 4  # 16th note duration
        
    def create_pattern(self, steps, sounds, pattern):
        """
        Create a sequence from a pattern
        
        Args:
            steps: Number of steps (usually 8 or 16)
            sounds: List of audio samples for each track
            pattern: 2D list - pattern[track][step] = velocity (0.0-1.0, 0=off)
        """
        sequence_duration = steps * self.step_duration
        total_samples = int(sequence_duration * self.sample_rate)
        output = np.zeros(total_samples)
        
        for track_idx, track_pattern in enumerate(pattern):
            if track_idx >= len(sounds):
                continue
                
            sound = sounds[track_idx]
            
            for step in range(min(steps, len(track_pattern))):
                velocity = track_pattern[step]
                if velocity > 0:
                    # Calculate start time for this step
                    start_time = step * self.step_duration
                    start_sample = int(start_time * self.sample_rate)
                    
                    # Scale sound by velocity and add to output
                    scaled_sound = sound * velocity
                    end_sample = min(start_sample + len(scaled_sound), total_samples)
                    actual_length = end_sample - start_sample
                    
                    if actual_length > 0:
                        output[start_sample:end_sample] += scaled_sound[:actual_length]
        
        return output

class SynthwaveComposer:
    """High-level composer for creating synthwave tracks"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.sequencer = StepSequencer(sample_rate)
        
    def create_bass_sound(self, frequency, duration=0.5):
        """Create a synthwave bass hit"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Sawtooth wave
        sawtooth = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        
        # Envelope (punchy)
        envelope = np.exp(-t * 8)  # Quick decay
        
        # Simple low-pass filtering effect
        filtered = sawtooth * envelope * 0.6
        
        return filtered
    
    def create_lead_sound(self, frequency, duration=0.25):
        """Create a synthwave lead hit"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Square wave
        sine = np.sin(2 * np.pi * frequency * t)
        square = np.sign(sine)
        
        # Sustained envelope
        envelope = np.exp(-t * 3)
        
        return square * envelope * 0.4
    
    def create_pad_chord(self, frequencies, duration=2.0):
        """Create a sustained pad chord"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        chord_sound = np.zeros_like(t)
        for freq in frequencies:
            # Use triangle waves for smooth pad sound
            sawtooth = 2 * (t * freq - np.floor(t * freq + 0.5))
            triangle = 2 * np.abs(sawtooth) - 1
            chord_sound += triangle
        
        # Normalize
        chord_sound /= len(frequencies)
        
        # Slow attack envelope
        attack_time = 0.5
        attack_samples = int(attack_time * self.sample_rate)
        envelope = np.ones_like(t)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        return chord_sound * envelope * 0.3
    
    def create_drum_kit(self):
        """Create basic drum sounds"""
        drums = {}
        
        # Kick drum - low frequency sine with quick decay
        kick_duration = 0.3
        t = np.linspace(0, kick_duration, int(self.sample_rate * kick_duration), False)
        kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 10)
        drums['kick'] = kick * 0.8
        
        # Snare - mix of noise and tone
        snare_duration = 0.2
        t = np.linspace(0, snare_duration, int(self.sample_rate * snare_duration), False)
        noise = np.random.normal(0, 0.1, len(t))
        tone = np.sin(2 * np.pi * 200 * t)
        snare = (noise + tone * 0.5) * np.exp(-t * 15)
        drums['snare'] = snare * 0.6
        
        # Hi-hat - filtered noise
        hihat_duration = 0.1
        t = np.linspace(0, hihat_duration, int(self.sample_rate * hihat_duration), False)
        noise = np.random.normal(0, 0.05, len(t))
        hihat = noise * np.exp(-t * 30)
        drums['hihat'] = hihat * 0.4
        
        return drums

def create_synthwave_demo():
    """Create a complete synthwave demo track"""
    composer = SynthwaveComposer()
    
    # Set up musical elements
    scale_freqs = MusicalScale.get_scale('A', 'minor', octave=3)  # Bass octave
    lead_freqs = MusicalScale.get_scale('A', 'minor', octave=5)   # Lead octave
    
    # Create chord progression (i - VI - III - VII in A minor)
    chords = [
        MusicalScale.get_chord('A', 'minor', octave=3),    # Am
        MusicalScale.get_chord('F', 'major', octave=3),    # F
        MusicalScale.get_chord('C', 'major', octave=3),    # C
        MusicalScale.get_chord('G', 'major', octave=3),    # G
    ]
    
    # Create sounds
    bass_sounds = [composer.create_bass_sound(freq) for freq in scale_freqs[:4]]
    lead_sounds = [composer.create_lead_sound(freq) for freq in lead_freqs[:4]]
    drums = composer.create_drum_kit()
    
    # Create patterns (8 steps each)
    bass_pattern = [
        [1.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0],  # Bass line
    ]
    
    lead_pattern = [
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.6, 0.0, 0.4],  # Lead melody
    ]
    
    drum_pattern = [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Kick
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Snare  
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],  # Hi-hat
    ]
    
    # Generate sequences
    bass_seq = composer.sequencer.create_pattern(8, bass_sounds, bass_pattern)
    lead_seq = composer.sequencer.create_pattern(8, lead_sounds, lead_pattern)
    drum_seq = composer.sequencer.create_pattern(8, [drums['kick'], drums['snare'], drums['hihat']], drum_pattern)
    
    # Create pad chord (sustained)
    pad = composer.create_pad_chord(chords[0], duration=len(bass_seq) / composer.sample_rate)
    
    # Mix everything together
    # Ensure all sequences are the same length
    max_length = max(len(bass_seq), len(lead_seq), len(drum_seq), len(pad))
    
    def pad_to_length(audio, target_length):
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)))
        return audio[:target_length]
    
    bass_seq = pad_to_length(bass_seq, max_length)
    lead_seq = pad_to_length(lead_seq, max_length)
    drum_seq = pad_to_length(drum_seq, max_length)
    pad = pad_to_length(pad, max_length)
    
    # Final mix
    final_mix = bass_seq + lead_seq + drum_seq + pad
    
    # Normalize to prevent clipping
    final_mix = final_mix / np.max(np.abs(final_mix)) * 0.8
    
    return final_mix

def main():
    """Demonstrate sequencing and composition"""
    print("=== Sequencer and Synthwave Composition ===")
    
    # Demonstrate musical scales
    print("1. Musical Scales and Frequencies")
    a_minor = MusicalScale.get_scale('A', 'minor')
    print(f"A Minor scale frequencies: {[f'{f:.1f} Hz' for f in a_minor]}")
    
    # Demonstrate chords
    am_chord = MusicalScale.get_chord('A', 'minor')
    print(f"A Minor chord frequencies: {[f'{f:.1f} Hz' for f in am_chord]}")
    
    # Create and play a simple sequence
    print("\n2. Creating drum sequence...")
    composer = SynthwaveComposer()
    drums = composer.create_drum_kit()
    
    # Simple drum pattern
    drum_pattern = [
        [1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0],  # Kick
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Snare
    ]
    
    drum_seq = composer.sequencer.create_pattern(
        8, [drums['kick'], drums['snare']], drum_pattern
    )
    
    print("Playing drum sequence...")
    sd.play(drum_seq, composer.sample_rate)
    sd.wait()
    
    # Create full synthwave demo
    print("\n3. Creating complete synthwave demo...")
    full_track = create_synthwave_demo()
    
    print("Playing synthwave demo (bass + lead + drums + pad)...")
    sd.play(full_track, composer.sample_rate)
    sd.wait()
    
    # Save the demo
    sf.write("synthwave_demo.wav", full_track, composer.sample_rate)
    print("Saved complete demo as 'synthwave_demo.wav'")
    
    print("\n=== Composition Concepts Demonstrated ===")
    print("• Step Sequencing: Programming rhythmic patterns")
    print("• Musical Scales: Mathematical relationships between notes")
    print("• Chord Progressions: Harmonic movement")
    print("• Layering: Combining multiple synthesizer parts")
    print("• Mixing: Balancing levels and avoiding clipping")
    print("• Sound Design: Creating characteristic synthwave timbres")

if __name__ == "__main__":
    main() 