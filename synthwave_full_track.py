"""
Full Synthwave Track Generator

This creates a complete synthwave track using the patterns and sounds
from the sequencer and composition module, expanded into a full song structure.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import importlib.util
import sys

# Import from 04_sequencer_and_composition.py
spec = importlib.util.spec_from_file_location("sequencer_module", "04_sequencer_and_composition.py")
sequencer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequencer_module)

MusicalScale = sequencer_module.MusicalScale
StepSequencer = sequencer_module.StepSequencer
SynthwaveComposer = sequencer_module.SynthwaveComposer

class FullTrackComposer(SynthwaveComposer):
    """Extended composer for creating full-length tracks"""
    
    def __init__(self, sample_rate=44100, bpm=120):
        super().__init__(sample_rate)
        self.bpm = bpm
        self.sequencer = StepSequencer(sample_rate, bpm)
        
    def create_section(self, patterns, sounds, bars=4, steps_per_bar=8):
        """Create a musical section with multiple bars"""
        total_steps = bars * steps_per_bar
        sections = []
        
        for bar in range(bars):
            bar_audio = self.sequencer.create_pattern(steps_per_bar, sounds, patterns)
            sections.append(bar_audio)
        
        return np.concatenate(sections)
    
    def create_variation_pattern(self, base_pattern, variation_type='fills'):
        """Create pattern variations for interest"""
        pattern = [row[:] for row in base_pattern]  # Deep copy
        
        if variation_type == 'fills':
            # Add fills to last step
            for track in pattern:
                if len(track) >= 8:
                    track[7] = min(1.0, track[7] + 0.6)  # Add fill on last step
        
        elif variation_type == 'breakdown':
            # Reduce to just kicks and hi-hats
            for i, track in enumerate(pattern):
                if i == 1:  # Snare track
                    pattern[i] = [0.0] * len(track)
        
        elif variation_type == 'buildup':
            # Gradually increase hi-hat intensity
            for i, track in enumerate(pattern):
                if i == 2:  # Hi-hat track
                    for j in range(len(track)):
                        if track[j] > 0:
                            track[j] = min(1.0, track[j] * (1 + j * 0.1))
        
        return pattern
    
    def create_arpeggio_pattern(self, chord_freqs, steps=16):
        """Create an arpeggiated pattern from chord frequencies"""
        pattern = []
        for i in range(steps):
            note_idx = i % len(chord_freqs)
            velocity = 0.7 if i % 4 == 0 else 0.4  # Accent every 4th step
            step_pattern = [0.0] * steps
            step_pattern[i] = velocity
            pattern.append(step_pattern)
        
        return pattern

def create_full_synthwave_track():
    """Create a complete synthwave track with multiple sections"""
    
    composer = FullTrackComposer(bpm=128)  # Slightly faster tempo
    
    print("Generating synthwave track...")
    print("Building musical elements...")
    
    # Musical setup
    scale_freqs = MusicalScale.get_scale('A', 'minor', octave=3)
    lead_freqs = MusicalScale.get_scale('A', 'minor', octave=5)
    
    # Extended chord progression (8 chords for verse/chorus)
    chord_progression = [
        MusicalScale.get_chord('A', 'minor', octave=3),    # Am
        MusicalScale.get_chord('F', 'major', octave=3),    # F
        MusicalScale.get_chord('C', 'major', octave=3),    # C
        MusicalScale.get_chord('G', 'major', octave=3),    # G
        MusicalScale.get_chord('D', 'minor', octave=3),    # Dm
        MusicalScale.get_chord('B', 'minor', octave=2),    # Bm (low)
        MusicalScale.get_chord('E', 'major', octave=3),    # E
        MusicalScale.get_chord('A', 'minor', octave=3),    # Am
    ]
    
    # Create sound libraries
    bass_sounds = [composer.create_bass_sound(freq, 0.6) for freq in scale_freqs]
    lead_sounds = [composer.create_lead_sound(freq, 0.3) for freq in lead_freqs]
    drums = composer.create_drum_kit()
    
    # Core patterns
    bass_pattern_verse = [
        [1.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    ]
    
    bass_pattern_chorus = [
        [1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.7, 0.0, 1.0, 0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0],
    ]
    
    lead_pattern_verse = [
        [0.0, 0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.5, 0.0, 0.3],
    ]
    
    lead_pattern_chorus = [
        [0.8, 0.0, 0.0, 0.6, 0.0, 0.0, 0.9, 0.0, 0.7, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0],
    ]
    
    drum_pattern_verse = [
        [1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],  # Kick
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Snare
        [0.0, 0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 0.4],  # Hi-hat
    ]
    
    drum_pattern_chorus = [
        [1.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.4, 0.0, 1.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.5, 0.0],  # Kick
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Snare
        [0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.7, 0.7],  # Hi-hat
    ]
    
    # Track sections
    track_sections = []
    
    print("Creating intro...")
    # INTRO (8 bars) - Just drums and bass
    intro_drums = composer.create_section(drum_pattern_verse, [drums['kick'], drums['snare'], drums['hihat']], bars=4, steps_per_bar=16)
    intro_bass = composer.create_section(bass_pattern_verse, bass_sounds, bars=4, steps_per_bar=16)
    intro_pad = composer.create_pad_chord(chord_progression[0], duration=len(intro_drums) / composer.sample_rate)
    
    intro = intro_drums + intro_bass + intro_pad * 0.3
    track_sections.append(intro)
    
    print("Creating verse 1...")
    # VERSE 1 (8 bars) - Add lead
    verse1_drums = composer.create_section(drum_pattern_verse, [drums['kick'], drums['snare'], drums['hihat']], bars=8, steps_per_bar=16)
    verse1_bass = composer.create_section(bass_pattern_verse, bass_sounds, bars=8, steps_per_bar=16)
    verse1_lead = composer.create_section(lead_pattern_verse, lead_sounds, bars=8, steps_per_bar=16)
    
    # Create chord progression for verse
    verse_pads = []
    for i in range(8):
        chord_idx = i % len(chord_progression)
        chord_duration = len(verse1_drums) / composer.sample_rate / 8
        pad = composer.create_pad_chord(chord_progression[chord_idx], duration=chord_duration)
        verse_pads.append(pad)
    verse1_pad = np.concatenate(verse_pads)
    
    verse1 = verse1_drums + verse1_bass + verse1_lead + verse1_pad * 0.4
    track_sections.append(verse1)
    
    print("Creating chorus 1...")
    # CHORUS 1 (8 bars) - Full energy
    chorus1_drums = composer.create_section(drum_pattern_chorus, [drums['kick'], drums['snare'], drums['hihat']], bars=8, steps_per_bar=16)
    chorus1_bass = composer.create_section(bass_pattern_chorus, bass_sounds, bars=8, steps_per_bar=16)
    chorus1_lead = composer.create_section(lead_pattern_chorus, lead_sounds, bars=8, steps_per_bar=16)
    
    # Brighter chord voicings for chorus
    chorus_chords = [MusicalScale.get_chord(note, 'major', octave=4) for note in ['F', 'C', 'G', 'A']]
    chorus_pads = []
    for i in range(8):
        chord_idx = i % len(chorus_chords)
        chord_duration = len(chorus1_drums) / composer.sample_rate / 8
        pad = composer.create_pad_chord(chorus_chords[chord_idx], duration=chord_duration)
        chorus_pads.append(pad)
    chorus1_pad = np.concatenate(chorus_pads)
    
    chorus1 = chorus1_drums + chorus1_bass + chorus1_lead + chorus1_pad * 0.5
    track_sections.append(chorus1)
    
    print("Creating breakdown...")
    # BREAKDOWN (4 bars) - Stripped down
    breakdown_pattern = composer.create_variation_pattern(drum_pattern_verse, 'breakdown')
    breakdown_drums = composer.create_section(breakdown_pattern, [drums['kick'], drums['snare'], drums['hihat']], bars=4, steps_per_bar=16)
    breakdown_bass = composer.create_section(bass_pattern_verse, bass_sounds, bars=4, steps_per_bar=16) * 0.5
    breakdown_pad = composer.create_pad_chord(chord_progression[0], duration=len(breakdown_drums) / composer.sample_rate)
    
    breakdown = breakdown_drums + breakdown_bass + breakdown_pad * 0.6
    track_sections.append(breakdown)
    
    print("Creating buildup...")
    # BUILDUP (4 bars) - Rising energy
    buildup_pattern = composer.create_variation_pattern(drum_pattern_verse, 'buildup')
    buildup_drums = composer.create_section(buildup_pattern, [drums['kick'], drums['snare'], drums['hihat']], bars=4, steps_per_bar=16)
    buildup_bass = composer.create_section(bass_pattern_verse, bass_sounds, bars=4, steps_per_bar=16)
    buildup_lead = composer.create_section(lead_pattern_verse, lead_sounds, bars=4, steps_per_bar=16)
    
    # Rising filter effect (approximate with volume)
    buildup_lead = buildup_lead * np.linspace(0.3, 1.0, len(buildup_lead))
    
    buildup = buildup_drums + buildup_bass + buildup_lead
    track_sections.append(buildup)
    
    print("Creating final chorus...")
    # FINAL CHORUS (8 bars) - Maximum energy
    final_drums = composer.create_section(drum_pattern_chorus, [drums['kick'], drums['snare'], drums['hihat']], bars=8, steps_per_bar=16)
    final_bass = composer.create_section(bass_pattern_chorus, bass_sounds, bars=8, steps_per_bar=16)
    final_lead = composer.create_section(lead_pattern_chorus, lead_sounds, bars=8, steps_per_bar=16)
    
    # Add variation to final chorus
    fill_pattern = composer.create_variation_pattern(drum_pattern_chorus, 'fills')
    fill_drums = composer.create_section(fill_pattern, [drums['kick'], drums['snare'], drums['hihat']], bars=2, steps_per_bar=16)
    
    final_section = final_drums + final_bass + final_lead * 1.2
    track_sections.append(final_section)
    
    print("Creating outro...")
    # OUTRO (4 bars) - Fade out
    outro_drums = composer.create_section(drum_pattern_verse, [drums['kick'], drums['snare'], drums['hihat']], bars=4, steps_per_bar=16)
    outro_bass = composer.create_section(bass_pattern_verse, bass_sounds, bars=4, steps_per_bar=16)
    outro_pad = composer.create_pad_chord(chord_progression[0], duration=len(outro_drums) / composer.sample_rate)
    
    outro = (outro_drums + outro_bass + outro_pad * 0.5) * np.linspace(1.0, 0.1, len(outro_drums))
    track_sections.append(outro)
    
    print("Mixing final track...")
    # Combine all sections
    full_track = np.concatenate(track_sections)
    
    # Master compression/limiting (simple)
    full_track = np.tanh(full_track * 0.8) * 0.9  # Soft clipping
    
    # Normalize
    full_track = full_track / np.max(np.abs(full_track)) * 0.85
    
    return full_track

def main():
    """Generate and play the full synthwave track"""
    print("=== Full Synthwave Track Generator ===")
    print("Building on the patterns from 04_sequencer_and_composition.py")
    print("Creating a complete track with intro, verse, chorus, breakdown, and outro...")
    
    # Generate the track
    track = create_full_synthwave_track()
    
    # Calculate duration
    sample_rate = 44100
    duration = len(track) / sample_rate
    print(f"\nTrack completed! Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Save the track
    filename = "full_synthwave_track.wav"
    sf.write(filename, track, sample_rate)
    print(f"Saved as: {filename}")
    
    # Play the track
    print("\nPlaying full synthwave track...")
    print("Track structure:")
    print("• Intro (drums + bass + pad)")
    print("• Verse 1 (+ lead melody)")
    print("• Chorus 1 (full energy)")
    print("• Breakdown (stripped down)")
    print("• Buildup (rising energy)")
    print("• Final Chorus (maximum energy)")
    print("• Outro (fade out)")
    
    sd.play(track, sample_rate)
    sd.wait()
    
    print("\nTrack completed! 🎵")

if __name__ == "__main__":
    main() 