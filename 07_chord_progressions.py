"""
Chord Progression Generator - Music Theory in Action

This module demonstrates:
- Music theory fundamentals (scales, chords, progressions)
- Chord voicings and inversions
- Popular progression patterns
- Harmonic rhythm and voice leading
- Chord synthesis and arrangement
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import random

class MusicTheory:
    def __init__(self):
        # Musical intervals in semitones
        self.intervals = {
            'unison': 0, 'minor_2nd': 1, 'major_2nd': 2, 'minor_3rd': 3,
            'major_3rd': 4, 'perfect_4th': 5, 'tritone': 6, 'perfect_5th': 7,
            'minor_6th': 8, 'major_6th': 9, 'minor_7th': 10, 'major_7th': 11, 'octave': 12
        }
        
        # Scale patterns (intervals from root)
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'pentatonic_major': [0, 2, 4, 7, 9],
            'pentatonic_minor': [0, 3, 5, 7, 10],
            'blues': [0, 3, 5, 6, 7, 10]
        }
        
        # Chord patterns (intervals from root)
        self.chord_types = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'major7': [0, 4, 7, 11],
            'minor7': [0, 3, 7, 10],
            'dominant7': [0, 4, 7, 10],
            'major9': [0, 4, 7, 11, 14],
            'minor9': [0, 3, 7, 10, 14],
            'suspended4': [0, 5, 7],
            'suspended2': [0, 2, 7],
            'add9': [0, 4, 7, 14]
        }
        
        # Note names
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Reference frequency (A4 = 440 Hz)
        self.A4_freq = 440.0
        self.A4_midi = 69
        
    def midi_to_freq(self, midi_note):
        """Convert MIDI note number to frequency"""
        return self.A4_freq * (2 ** ((midi_note - self.A4_midi) / 12))
    
    def note_name_to_midi(self, note_name, octave=4):
        """Convert note name to MIDI number"""
        note_offset = self.note_names.index(note_name)
        return (octave + 1) * 12 + note_offset
    
    def get_scale_notes(self, root_note, scale_type, octave=4):
        """Get MIDI note numbers for a scale"""
        root_midi = self.note_name_to_midi(root_note, octave)
        scale_pattern = self.scales[scale_type]
        return [root_midi + interval for interval in scale_pattern]
    
    def get_chord_notes(self, root_note, chord_type, octave=4, inversion=0):
        """Get MIDI note numbers for a chord"""
        root_midi = self.note_name_to_midi(root_note, octave)
        chord_pattern = self.chord_types[chord_type]
        notes = [root_midi + interval for interval in chord_pattern]
        
        # Apply inversion
        for _ in range(inversion):
            notes.append(notes.pop(0) + 12)  # Move bottom note up an octave
        
        return notes
    
    def get_chord_name(self, root_note, chord_type):
        """Get the full chord name"""
        chord_symbols = {
            'major': '', 'minor': 'm', 'diminished': 'dim', 'augmented': 'aug',
            'major7': 'maj7', 'minor7': 'm7', 'dominant7': '7',
            'major9': 'maj9', 'minor9': 'm9',
            'suspended4': 'sus4', 'suspended2': 'sus2', 'add9': 'add9'
        }
        return root_note + chord_symbols[chord_type]

class ChordSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.theory = MusicTheory()
    
    def generate_chord_tone(self, frequency, duration, waveform='sine', volume=0.3):
        """Generate a single chord tone"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        if waveform == 'sine':
            wave = np.sin(2 * np.pi * frequency * t)
        elif waveform == 'square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform == 'sawtooth':
            wave = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        elif waveform == 'triangle':
            square = np.sign(np.sin(2 * np.pi * frequency * t))
            # Approximate triangle by integrating square wave
            wave = np.cumsum(square)
            wave = wave / np.max(np.abs(wave))  # Normalize
        
        # Apply ADSR envelope
        attack = 0.1
        decay = 0.2
        sustain_level = 0.7
        release = 0.3
        
        envelope = self.adsr_envelope(duration, attack, decay, sustain_level, release)
        return wave * envelope * volume
    
    def adsr_envelope(self, duration, attack, decay, sustain_level, release):
        """Generate ADSR envelope"""
        total_samples = int(duration * self.sample_rate)
        envelope = np.zeros(total_samples)
        
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        current_sample = 0
        
        # Attack
        if attack_samples > 0:
            envelope[current_sample:current_sample + attack_samples] = np.linspace(0, 1, attack_samples)
            current_sample += attack_samples
        
        # Decay
        if decay_samples > 0:
            envelope[current_sample:current_sample + decay_samples] = np.linspace(1, sustain_level, decay_samples)
            current_sample += decay_samples
        
        # Sustain
        if sustain_samples > 0:
            envelope[current_sample:current_sample + sustain_samples] = sustain_level
            current_sample += sustain_samples
        
        # Release
        if release_samples > 0 and current_sample < total_samples:
            remaining = total_samples - current_sample
            envelope[current_sample:current_sample + remaining] = np.linspace(sustain_level, 0, remaining)
        
        return envelope
    
    def synthesize_chord(self, midi_notes, duration, waveform='sine', spread=0.02):
        """Synthesize a full chord"""
        chord_audio = np.zeros(int(duration * self.sample_rate))
        
        for i, midi_note in enumerate(midi_notes):
            frequency = self.theory.midi_to_freq(midi_note)
            
            # Slight detuning for richness
            detune = (random.random() - 0.5) * spread
            frequency *= (1 + detune)
            
            tone = self.generate_chord_tone(frequency, duration, waveform)
            
            # Pan slightly for width
            if len(midi_notes) > 1:
                pan = (i / (len(midi_notes) - 1)) * 2 - 1  # -1 to 1
                left_gain = (1 - pan) / 2
                right_gain = (1 + pan) / 2
                # For mono output, just add all tones
                chord_audio += tone
            else:
                chord_audio += tone
        
        # Normalize
        if np.max(np.abs(chord_audio)) > 0:
            chord_audio = chord_audio / np.max(np.abs(chord_audio)) * 0.8
        
        return chord_audio

class ProgressionGenerator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.theory = MusicTheory()
        self.synth = ChordSynthesizer(sample_rate)
        
        # Popular chord progressions (using scale degrees)
        self.common_progressions = {
            'I-V-vi-IV': [1, 5, 6, 4],  # C-G-Am-F (very common)
            'vi-IV-I-V': [6, 4, 1, 5],  # Am-F-C-G  
            'I-vi-IV-V': [1, 6, 4, 5],  # C-Am-F-G (50s progression)
            'ii-V-I': [2, 5, 1],        # Dm-G-C (jazz standard)
            'I-VII-IV-I': [1, 7, 4, 1], # C-Bb-F-C (modal)
            'I-bVII-IV': [1, 7, 4],     # C-Bb-F (rock progression)
            'vi-bVII-I': [6, 7, 1],     # Am-Bb-C (modern pop)
        }
    
    def scale_degree_to_chord(self, degree, key, scale_type='major', chord_quality=None):
        """Convert scale degree to chord in the given key"""
        scale_notes = self.theory.get_scale_notes(key, scale_type, octave=4)
        
        # Get the root note for this degree
        root_midi = scale_notes[(degree - 1) % len(scale_notes)]
        
        # Determine chord quality if not specified
        if chord_quality is None:
            if scale_type == 'major':
                # Major scale chord qualities: I ii iii IV V vi vii°
                qualities = ['major', 'minor', 'minor', 'major', 'major', 'minor', 'diminished']
                chord_quality = qualities[(degree - 1) % 7]
            elif scale_type == 'minor':
                # Natural minor chord qualities: i ii° III iv v VI VII
                qualities = ['minor', 'diminished', 'major', 'minor', 'minor', 'major', 'major']
                chord_quality = qualities[(degree - 1) % 7]
        
        # Get chord notes
        root_note_name = self.theory.note_names[root_midi % 12]
        chord_notes = self.theory.get_chord_notes(root_note_name, chord_quality, octave=4)
        
        return chord_notes, f"{root_note_name}{self.theory.chord_types[chord_quality]}"
    
    def generate_progression(self, progression_name, key='C', scale_type='major', 
                           chord_duration=2.0, waveform='sine'):
        """Generate audio for a chord progression"""
        if progression_name not in self.common_progressions:
            raise ValueError(f"Unknown progression: {progression_name}")
        
        degrees = self.common_progressions[progression_name]
        chord_audio_list = []
        chord_names = []
        
        for degree in degrees:
            chord_notes, chord_name = self.scale_degree_to_chord(degree, key, scale_type)
            chord_audio = self.synth.synthesize_chord(chord_notes, chord_duration, waveform)
            chord_audio_list.append(chord_audio)
            chord_names.append(chord_name)
        
        # Concatenate all chords
        full_progression = np.concatenate(chord_audio_list)
        
        return full_progression, chord_names
    
    def create_custom_progression(self, chord_specs, chord_duration=2.0, waveform='sine'):
        """Create progression from custom chord specifications
        chord_specs: List of (root_note, chord_type, octave) tuples
        """
        chord_audio_list = []
        chord_names = []
        
        for root_note, chord_type, octave in chord_specs:
            chord_notes = self.theory.get_chord_notes(root_note, chord_type, octave)
            chord_audio = self.synth.synthesize_chord(chord_notes, chord_duration, waveform)
            chord_audio_list.append(chord_audio)
            chord_names.append(self.theory.get_chord_name(root_note, chord_type))
        
        full_progression = np.concatenate(chord_audio_list)
        return full_progression, chord_names
    
    def add_bass_line(self, progression_audio, chord_names, chord_duration=2.0):
        """Add a simple bass line to the progression"""
        bass_audio = np.zeros(len(progression_audio))
        
        for i, chord_name in enumerate(chord_names):
            start_sample = int(i * chord_duration * self.sample_rate)
            
            # Extract root note from chord name
            root_note = chord_name[0] if len(chord_name) > 0 else 'C'
            if len(chord_name) > 1 and chord_name[1] == '#':
                root_note += '#'
            
            # Get bass note (root in lower octave)
            bass_midi = self.theory.note_name_to_midi(root_note, octave=2)
            bass_freq = self.theory.midi_to_freq(bass_midi)
            
            # Generate bass tone
            bass_tone = self.synth.generate_chord_tone(bass_freq, chord_duration, 
                                                     waveform='sawtooth', volume=0.4)
            
            end_sample = min(start_sample + len(bass_tone), len(bass_audio))
            bass_audio[start_sample:end_sample] = bass_tone[:end_sample - start_sample]
        
        # Mix bass with progression
        return progression_audio + bass_audio

def main():
    """Demonstrate chord progression generation"""
    print("=== Chord Progression Generator ===")
    
    generator = ProgressionGenerator()
    
    # 1. Basic music theory demonstration
    print("1. Music Theory Demonstration")
    theory = MusicTheory()
    
    # Show C major scale
    c_major_notes = theory.get_scale_notes('C', 'major', octave=4)
    c_major_freqs = [theory.midi_to_freq(note) for note in c_major_notes]
    print(f"C Major scale notes (MIDI): {c_major_notes}")
    print(f"C Major scale frequencies: {[f'{f:.1f} Hz' for f in c_major_freqs]}")
    
    # Show some chords
    chords_to_show = [('C', 'major'), ('A', 'minor'), ('F', 'major'), ('G', 'dominant7')]
    for root, chord_type in chords_to_show:
        chord_notes = theory.get_chord_notes(root, chord_type, octave=4)
        chord_name = theory.get_chord_name(root, chord_type)
        print(f"{chord_name}: {chord_notes}")
    
    # 2. Generate popular progressions
    print("\n2. Popular Chord Progressions")
    
    progressions_to_demo = ['I-V-vi-IV', 'vi-IV-I-V', 'ii-V-I']
    
    for prog_name in progressions_to_demo:
        print(f"\nGenerating {prog_name} progression in C major...")
        progression_audio, chord_names = generator.generate_progression(
            prog_name, key='C', chord_duration=1.5, waveform='sine'
        )
        
        print(f"Chords: {' - '.join(chord_names)}")
        
        # Add bass line
        with_bass = generator.add_bass_line(progression_audio, chord_names, chord_duration=1.5)
        
        print(f"Playing {prog_name}...")
        sd.play(with_bass, generator.sample_rate)
        sd.wait()
        
        # Save to file
        filename = f"progression_{prog_name.replace('-', '_')}.wav"
        sf.write(filename, with_bass, generator.sample_rate)
    
    # 3. Custom jazz-influenced progression
    print("\n3. Custom Jazz-influenced Progression")
    custom_chords = [
        ('C', 'major7', 4),
        ('A', 'minor7', 4),
        ('D', 'minor7', 4),
        ('G', 'dominant7', 4),
        ('E', 'minor7', 4),
        ('A', 'minor7', 4),
        ('F', 'major7', 4),
        ('G', 'dominant7', 4)
    ]
    
    jazz_progression, jazz_chord_names = generator.create_custom_progression(
        custom_chords, chord_duration=1.0, waveform='sine'
    )
    
    jazz_with_bass = generator.add_bass_line(jazz_progression, jazz_chord_names, chord_duration=1.0)
    
    print(f"Jazz progression: {' - '.join(jazz_chord_names)}")
    print("Playing jazz progression...")
    sd.play(jazz_with_bass, generator.sample_rate)
    sd.wait()
    
    sf.write("jazz_progression.wav", jazz_with_bass, generator.sample_rate)
    
    # 4. Visualization
    print("\n4. Chord Progression Visualization")
    
    # Visualize the I-V-vi-IV progression
    prog_audio, prog_chords = generator.generate_progression('I-V-vi-IV', chord_duration=1.0)
    
    # Create time axis
    time_axis = np.linspace(0, len(prog_audio) / generator.sample_rate, len(prog_audio))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, prog_audio)
    plt.title('I-V-vi-IV Progression Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    
    # Add chord labels
    for i, chord in enumerate(prog_chords):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.7)
        plt.text(i + 0.5, 0.8, chord, fontsize=12, ha='center')
    
    plt.grid(True, alpha=0.3)
    
    # Frequency spectrum
    plt.subplot(2, 1, 2)
    fft = np.fft.fft(prog_audio)
    frequencies = np.fft.fftfreq(len(fft), 1/generator.sample_rate)
    magnitude = np.abs(fft)
    
    # Plot only positive frequencies up to 2000 Hz
    positive_freq_mask = (frequencies > 0) & (frequencies < 2000)
    plt.plot(frequencies[positive_freq_mask], magnitude[positive_freq_mask])
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Chord Progression Concepts Demonstrated ===")
    print("• Music Theory: Scales, chords, and progressions")
    print("• Roman Numeral Analysis: Understanding harmonic function")
    print("• Voice Leading: How chords connect smoothly")
    print("• Chord Voicings: Different ways to arrange chord tones")
    print("• Bass Lines: Foundation and harmonic rhythm")
    print("• Genre Styles: Pop, jazz, and rock progressions")

if __name__ == "__main__":
    main() 