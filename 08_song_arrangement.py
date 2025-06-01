"""
Song Arrangement and Structure - Building Complete Tracks

This module demonstrates:
- Song structure patterns (verse, chorus, bridge, etc.)
- Arrangement techniques and instrumentation
- Dynamic changes and build-ups
- Transitions and fills
- Complete track assembly and mixing
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from dataclasses import dataclass
from typing import List, Dict, Optional
import copy

# Import our previous modules for reuse
import sys
import importlib.util

# Load drum machine module
spec = importlib.util.spec_from_file_location("drum_machine", "06_drum_machine.py")
drum_machine_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drum_machine_module)
DrumMachine = drum_machine_module.DrumMachine
DrumSynthesizer = drum_machine_module.DrumSynthesizer

# Load chord progressions module  
spec = importlib.util.spec_from_file_location("chord_progressions", "07_chord_progressions.py")
chord_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chord_module)
ProgressionGenerator = chord_module.ProgressionGenerator
MusicTheory = chord_module.MusicTheory

# Load synthesizer module
spec = importlib.util.spec_from_file_location("synthesizer_basics", "02_synthesizer_basics.py")
synth_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth_module)
Synthesizer = synth_module.Synthesizer

@dataclass
class SongSection:
    """Represents a section of a song (verse, chorus, etc.)"""
    name: str
    duration: float  # in seconds
    chord_progression: List[str]  # chord names
    drum_pattern: str  # pattern identifier
    arrangement: Dict[str, Dict]  # instrument settings
    dynamics: float = 0.8  # volume level (0.0 to 1.0)
    tempo_modifier: float = 1.0  # tempo change multiplier

class SongArranger:
    def __init__(self, bpm=120, key='C', sample_rate=44100):
        self.bpm = bpm
        self.key = key
        self.sample_rate = sample_rate
        
        # Initialize our synthesis components
        self.drum_machine = DrumMachine(bpm, sample_rate)
        self.progression_gen = ProgressionGenerator(sample_rate)
        self.synth = Synthesizer(sample_rate)
        self.theory = MusicTheory()
        
        # Define common song structures
        self.song_structures = {
            'verse_chorus': ['intro', 'verse1', 'chorus1', 'verse2', 'chorus2', 'bridge', 'chorus3', 'outro'],
            'aaba': ['a1', 'a2', 'b', 'a3'],
            'pop_standard': ['intro', 'verse1', 'prechorus1', 'chorus1', 'verse2', 'prechorus2', 'chorus2', 'bridge', 'chorus3', 'outro'],
            'electronic': ['intro', 'buildup1', 'drop1', 'breakdown', 'buildup2', 'drop2', 'outro']
        }
        
                 # Define drum patterns for different sections
        self.drum_patterns = {
            'intro': {
                'kick': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'snare': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'hihat_closed': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            'verse': {
                'kick': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                'snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat_closed': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            'chorus': {
                'kick': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                'snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat_closed': [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'clap': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            },
            'bridge': {
                'kick': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                'snare': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'hihat_closed': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            'buildup': {
                'kick': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'snare': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                'hihat_closed': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        }
        
        # Define chord progressions for different sections
        self.section_progressions = {
            'verse': ['C', 'Am', 'F', 'G'],
            'chorus': ['F', 'C', 'G', 'Am'],
            'bridge': ['Am', 'F', 'C', 'G'],
            'intro': ['C'],
            'outro': ['F', 'C']
        }
    
    def create_bass_line(self, chord_names, duration_per_chord, pattern='root'):
        """Create bass line for chord progression"""
        total_duration = len(chord_names) * duration_per_chord
        bass_audio = np.zeros(int(total_duration * self.sample_rate))
        
        for i, chord_name in enumerate(chord_names):
            start_time = i * duration_per_chord
            start_sample = int(start_time * self.sample_rate)
            
            # Extract root note
            root_note = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] == '#':
                root_note += '#'
            
            # Create bass pattern
            if pattern == 'root':
                # Simple root note on beats
                bass_freq = self.theory.midi_to_freq(
                    self.theory.note_name_to_midi(root_note, octave=2)
                )
                bass_tone = self.synth.generate_sawtooth_wave(
                    bass_freq, duration_per_chord, amplitude=0.4
                )[1]
            elif pattern == 'walking':
                # Walking bass line (simplified)
                bass_freq = self.theory.midi_to_freq(
                    self.theory.note_name_to_midi(root_note, octave=2)
                )
                bass_tone = self.synth.generate_sawtooth_wave(
                    bass_freq, duration_per_chord, amplitude=0.4
                )[1]
            
            # Apply envelope
            envelope = self.synth.adsr_envelope(
                duration_per_chord, attack=0.01, decay=0.1, sustain_level=0.8, release=0.2
            )
            bass_tone *= envelope
            
            end_sample = min(start_sample + len(bass_tone), len(bass_audio))
            bass_audio[start_sample:end_sample] = bass_tone[:end_sample - start_sample]
        
        return bass_audio
    
    def create_lead_melody(self, chord_names, duration_per_chord, style='simple'):
        """Create lead melody over chord progression"""
        total_duration = len(chord_names) * duration_per_chord
        lead_audio = np.zeros(int(total_duration * self.sample_rate))
        
        # Define simple melody patterns
        melody_intervals = {
            'simple': [0, 2, 4, 2],  # root, 2nd, 3rd, 2nd
            'arpeggiated': [0, 4, 7, 4],  # root, 3rd, 5th, 3rd
            'scalar': [0, 2, 4, 5, 4, 2, 0, -1]  # scale run
        }
        
        pattern = melody_intervals.get(style, melody_intervals['simple'])
        
        for i, chord_name in enumerate(chord_names):
            start_time = i * duration_per_chord
            start_sample = int(start_time * self.sample_rate)
            
            # Get chord root
            root_note = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] == '#':
                root_note += '#'
            
            root_midi = self.theory.note_name_to_midi(root_note, octave=5)
            
            # Create melody notes for this chord
            note_duration = duration_per_chord / len(pattern)
            
            for j, interval in enumerate(pattern):
                note_start = start_time + j * note_duration
                note_start_sample = int(note_start * self.sample_rate)
                
                melody_note = root_midi + interval
                melody_freq = self.theory.midi_to_freq(melody_note)
                
                # Generate tone
                if style == 'arpeggiated':
                    tone = self.synth.create_synthwave_lead(melody_freq, note_duration)
                else:
                    tone = self.synth.generate_sine_wave(melody_freq, note_duration, amplitude=0.3)[1]
                    envelope = self.synth.adsr_envelope(
                        note_duration, attack=0.05, decay=0.1, sustain_level=0.6, release=0.2
                    )
                    tone *= envelope
                
                note_end_sample = min(note_start_sample + len(tone), len(lead_audio))
                lead_audio[note_start_sample:note_end_sample] += tone[:note_end_sample - note_start_sample]
        
        return lead_audio
    
    def create_pad_sound(self, chord_names, duration_per_chord):
        """Create atmospheric pad sound"""
        total_duration = len(chord_names) * duration_per_chord
        pad_audio = np.zeros(int(total_duration * self.sample_rate))
        
        for i, chord_name in enumerate(chord_names):
            start_time = i * duration_per_chord
            start_sample = int(start_time * self.sample_rate)
            
            # Get chord notes
            root_note = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] == '#':
                root_note += '#'
            
            # Create simple major/minor triad
            chord_type = 'minor' if 'm' in chord_name else 'major'
            chord_notes = self.theory.get_chord_notes(root_note, chord_type, octave=4)
            
            # Generate pad chord
            chord_audio = np.zeros(int(duration_per_chord * self.sample_rate))
            
            for note in chord_notes:
                freq = self.theory.midi_to_freq(note)
                # Use sine waves for smooth pad sound
                t, wave = self.synth.generate_sine_wave(freq, duration_per_chord, amplitude=0.15)
                
                # Long attack and release for pad
                envelope = self.synth.adsr_envelope(
                    duration_per_chord, attack=0.5, decay=0.3, sustain_level=0.8, release=0.8
                )
                wave *= envelope
                chord_audio += wave
            
            end_sample = min(start_sample + len(chord_audio), len(pad_audio))
            pad_audio[start_sample:end_sample] += chord_audio[:end_sample - start_sample]
        
        return pad_audio
    
    def apply_dynamics(self, audio, dynamics_curve):
        """Apply dynamic changes to audio"""
        if len(dynamics_curve) != len(audio):
            # Interpolate dynamics curve to match audio length
            x_old = np.linspace(0, 1, len(dynamics_curve))
            x_new = np.linspace(0, 1, len(audio))
            dynamics_curve = np.interp(x_new, x_old, dynamics_curve)
        
        return audio * dynamics_curve
    
    def create_section_audio(self, section: SongSection):
        """Generate audio for a complete song section"""
        print(f"Creating section: {section.name}")
        
        # Get drum pattern
        drum_pattern_name = section.drum_pattern
        if drum_pattern_name in self.drum_patterns:
            drum_pattern = self.drum_patterns[drum_pattern_name]
        else:
            drum_pattern = self.drum_patterns['verse']  # default
        
        # Calculate timing
        chord_duration = section.duration / len(section.chord_progression)
        
        # Generate drums
        drums_audio = self.drum_machine.sequence_pattern(drum_pattern)
        # Repeat/trim drums to match section duration
        target_samples = int(section.duration * self.sample_rate)
        if len(drums_audio) > target_samples:
            drums_audio = drums_audio[:target_samples]
        else:
            # Repeat pattern if needed
            repeats = int(np.ceil(target_samples / len(drums_audio)))
            drums_audio = np.tile(drums_audio, repeats)[:target_samples]
        
        # Generate chord progression
        chord_audio = self.progression_gen.synth.synthesize_chord(
            self.theory.get_chord_notes('C', 'major', octave=4), 
            section.duration, waveform='sine'
        )
        if section.chord_progression:
            # Create proper chord progression
            prog_audio = np.zeros(target_samples)
            for i, chord_name in enumerate(section.chord_progression):
                start_sample = int(i * chord_duration * self.sample_rate)
                
                root_note = chord_name[0]
                if len(chord_name) > 1 and chord_name[1] == '#':
                    root_note += '#'
                
                chord_type = 'minor' if 'm' in chord_name else 'major'
                chord_notes = self.theory.get_chord_notes(root_note, chord_type, octave=4)
                chord_tone = self.progression_gen.synth.synthesize_chord(
                    chord_notes, chord_duration, waveform='sine'
                )
                
                end_sample = min(start_sample + len(chord_tone), target_samples)
                prog_audio[start_sample:end_sample] = chord_tone[:end_sample - start_sample]
            
            chord_audio = prog_audio
        
        # Generate bass line
        bass_audio = self.create_bass_line(section.chord_progression, chord_duration)
        if len(bass_audio) > target_samples:
            bass_audio = bass_audio[:target_samples]
        elif len(bass_audio) < target_samples:
            bass_audio = np.pad(bass_audio, (0, target_samples - len(bass_audio)))
        
        # Generate additional elements based on arrangement
        elements = {'drums': drums_audio * 0.8, 'chords': chord_audio * 0.6, 'bass': bass_audio * 0.7}
        
        if 'lead' in section.arrangement:
            lead_audio = self.create_lead_melody(section.chord_progression, chord_duration, 
                                               section.arrangement['lead'].get('style', 'simple'))
            if len(lead_audio) > target_samples:
                lead_audio = lead_audio[:target_samples]
            elif len(lead_audio) < target_samples:
                lead_audio = np.pad(lead_audio, (0, target_samples - len(lead_audio)))
            elements['lead'] = lead_audio * 0.5
        
        if 'pad' in section.arrangement:
            pad_audio = self.create_pad_sound(section.chord_progression, chord_duration)
            if len(pad_audio) > target_samples:
                pad_audio = pad_audio[:target_samples]
            elif len(pad_audio) < target_samples:
                pad_audio = np.pad(pad_audio, (0, target_samples - len(pad_audio)))
            elements['pad'] = pad_audio * 0.4
        
        # Mix all elements
        mixed_audio = np.zeros(target_samples)
        for element_name, element_audio in elements.items():
            mixed_audio += element_audio
        
        # Apply section dynamics
        mixed_audio *= section.dynamics
        
        # Normalize to prevent clipping
        if np.max(np.abs(mixed_audio)) > 0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.8
        
        return mixed_audio
    
    def arrange_song(self, structure_name='verse_chorus', total_duration=60):
        """Create a complete song arrangement"""
        if structure_name not in self.song_structures:
            raise ValueError(f"Unknown song structure: {structure_name}")
        
        structure = self.song_structures[structure_name]
        section_duration = total_duration / len(structure)
        
        sections = []
        
        for section_name in structure:
            # Define section properties
            if 'intro' in section_name:
                chord_prog = self.section_progressions['intro']
                drum_pattern = 'intro'
                arrangement = {}
                dynamics = 0.5
            elif 'verse' in section_name:
                chord_prog = self.section_progressions['verse']
                drum_pattern = 'verse'
                arrangement = {'lead': {'style': 'simple'}}
                dynamics = 0.7
            elif 'chorus' in section_name:
                chord_prog = self.section_progressions['chorus']
                drum_pattern = 'chorus'
                arrangement = {'lead': {'style': 'arpeggiated'}, 'pad': {}}
                dynamics = 1.0
            elif 'bridge' in section_name:
                chord_prog = self.section_progressions['bridge']
                drum_pattern = 'bridge'
                arrangement = {'pad': {}}
                dynamics = 0.8
            elif 'outro' in section_name:
                chord_prog = self.section_progressions['outro']
                drum_pattern = 'intro'
                arrangement = {}
                dynamics = 0.6
            else:
                # Default section
                chord_prog = self.section_progressions['verse']
                drum_pattern = 'verse'
                arrangement = {}
                dynamics = 0.7
            
            section = SongSection(
                name=section_name,
                duration=section_duration,
                chord_progression=chord_prog,
                drum_pattern=drum_pattern,
                arrangement=arrangement,
                dynamics=dynamics
            )
            sections.append(section)
        
        return sections
    
    def render_song(self, sections: List[SongSection]):
        """Render complete song from sections"""
        section_audio_list = []
        
        for section in sections:
            section_audio = self.create_section_audio(section)
            section_audio_list.append(section_audio)
        
        # Concatenate all sections
        full_song = np.concatenate(section_audio_list)
        
        # Final mastering (simple compression and limiting)
        full_song = self.simple_compressor(full_song)
        
        return full_song, sections
    
    def simple_compressor(self, audio, threshold=0.7, ratio=3.0):
        """Apply simple compression to audio"""
        compressed = np.copy(audio)
        
        # Find samples above threshold
        above_threshold = np.abs(compressed) > threshold
        
        # Apply compression
        compressed[above_threshold] = np.sign(compressed[above_threshold]) * (
            threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio
        )
        
        return compressed

def main():
    """Demonstrate song arrangement and structure"""
    print("=== Song Arrangement and Structure ===")
    
    arranger = SongArranger(bpm=120, key='C')
    
    # 1. Create different song structures
    print("1. Creating Song Sections")
    
    # Create a verse-chorus song
    sections = arranger.arrange_song('verse_chorus', total_duration=40)
    
    print("Song Structure:")
    for i, section in enumerate(sections):
        print(f"{i+1}. {section.name}: {section.duration:.1f}s - {section.chord_progression} - dynamics: {section.dynamics}")
    
    # 2. Render the complete song
    print("\n2. Rendering Complete Song")
    full_song, rendered_sections = arranger.render_song(sections)
    
    print(f"Total song duration: {len(full_song) / arranger.sample_rate:.1f} seconds")
    print("Playing complete song...")
    sd.play(full_song, arranger.sample_rate)
    sd.wait()
    
    # Save the song
    sf.write("complete_song.wav", full_song, arranger.sample_rate)
    print("Saved complete song as 'complete_song.wav'")
    
    # 3. Create shorter demo sections
    print("\n3. Creating Individual Section Demos")
    
    # Demo verse
    verse_section = SongSection(
        name="verse_demo",
        duration=8.0,
        chord_progression=['C', 'Am', 'F', 'G'],
        drum_pattern='verse',
        arrangement={'lead': {'style': 'simple'}},
        dynamics=0.8
    )
    
    verse_audio = arranger.create_section_audio(verse_section)
    print("Playing verse section...")
    sd.play(verse_audio, arranger.sample_rate)
    sd.wait()
    
    # Demo chorus
    chorus_section = SongSection(
        name="chorus_demo",
        duration=8.0,
        chord_progression=['F', 'C', 'G', 'Am'],
        drum_pattern='chorus',
        arrangement={'lead': {'style': 'arpeggiated'}, 'pad': {}},
        dynamics=1.0
    )
    
    chorus_audio = arranger.create_section_audio(chorus_section)
    print("Playing chorus section...")
    sd.play(chorus_audio, arranger.sample_rate)
    sd.wait()
    
    # 4. Visualization
    print("\n4. Song Structure Visualization")
    
    # Create time axis for visualization
    time_axis = np.linspace(0, len(full_song) / arranger.sample_rate, len(full_song))
    
    plt.figure(figsize=(15, 8))
    
    # Waveform
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, full_song)
    plt.title('Complete Song Waveform')
    plt.ylabel('Amplitude')
    
    # Add section markers
    current_time = 0
    for section in rendered_sections:
        plt.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
        plt.text(current_time + section.duration/2, 0.8, section.name, 
                fontsize=10, ha='center', rotation=45)
        current_time += section.duration
    
    plt.grid(True, alpha=0.3)
    
    # Dynamic levels
    plt.subplot(3, 1, 2)
    section_times = []
    section_dynamics = []
    current_time = 0
    
    for section in rendered_sections:
        section_times.extend([current_time, current_time + section.duration])
        section_dynamics.extend([section.dynamics, section.dynamics])
        current_time += section.duration
    
    plt.plot(section_times, section_dynamics, linewidth=2)
    plt.title('Dynamic Levels Throughout Song')
    plt.ylabel('Dynamics')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Frequency spectrum
    plt.subplot(3, 1, 3)
    fft = np.fft.fft(full_song)
    frequencies = np.fft.fftfreq(len(fft), 1/arranger.sample_rate)
    magnitude = np.abs(fft)
    
    positive_freq_mask = (frequencies > 0) & (frequencies < 5000)
    plt.plot(frequencies[positive_freq_mask], magnitude[positive_freq_mask])
    plt.title('Overall Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Song Arrangement Concepts Demonstrated ===")
    print("• Song Structure: Verse-Chorus-Bridge arrangements")
    print("• Instrumentation: Layering different synthesized elements")
    print("• Dynamics: Building and releasing energy through sections")
    print("• Arrangement: Adding/removing elements for variety")
    print("• Mixing: Balancing levels between instruments")
    print("• Song Form: Creating musical narrative and flow")

if __name__ == "__main__":
    main() 