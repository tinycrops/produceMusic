"""
Complete Music Production Demo - Bringing It All Together

This program demonstrates all the concepts from the music production series:
- Digital audio fundamentals
- Synthesizer basics (oscillators, envelopes, LFOs)
- Effects and processing (reverb, delay, distortion)
- Drum machine and rhythmic programming  
- Chord progressions and music theory
- Song arrangement and structure

It creates a complete, professionally arranged electronic music track.
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import sys
import importlib.util

def load_module(name, file_path):
    """Helper function to load modules from files"""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load all our music production modules
audio_module = load_module("digital_audio", "01_digital_audio_basics.py")
synth_module = load_module("synthesizer", "02_synthesizer_basics.py")
effects_module = load_module("effects", "03_effects_and_processing.py")
sequencer_module = load_module("sequencer", "04_sequencer_and_composition.py")
drum_module = load_module("drums", "06_drum_machine.py")
chord_module = load_module("chords", "07_chord_progressions.py")
arrangement_module = load_module("arrangement", "08_song_arrangement.py")

class CompleteMusicProducer:
    def __init__(self, bpm=128, key='C', sample_rate=44100):
        self.bpm = bpm
        self.key = key
        self.sample_rate = sample_rate
        
        # Initialize all our components
        self.synthesizer = synth_module.Synthesizer(sample_rate)
        self.drum_machine = drum_module.DrumMachine(bpm, sample_rate)
        self.progression_gen = chord_module.ProgressionGenerator(sample_rate)
        self.effects = effects_module.AudioEffects(sample_rate)
        self.arranger = arrangement_module.SongArranger(bpm, key, sample_rate)
        self.theory = chord_module.MusicTheory()
        
    def create_professional_track(self):
        """Create a complete professional-sounding electronic music track"""
        print("=== Creating Professional Electronic Music Track ===")
        
        # Define the song structure
        sections = []
        
        # 1. Intro (8 seconds) - Atmospheric pad with filtered drums
        print("Creating intro section...")
        intro_chords = ['C', 'Am']
        intro_audio = self.create_intro(intro_chords, 8.0)
        sections.append(('Intro', intro_audio))
        
        # 2. Verse 1 (16 seconds) - Bass, drums, subtle lead
        print("Creating verse 1...")
        verse_chords = ['C', 'Am', 'F', 'G']
        verse1_audio = self.create_verse(verse_chords, 16.0, energy_level=0.6)
        sections.append(('Verse 1', verse1_audio))
        
        # 3. Build-up (8 seconds) - Rising tension
        print("Creating build-up...")
        buildup_audio = self.create_buildup(verse_chords, 8.0)
        sections.append(('Build-up', buildup_audio))
        
        # 4. Chorus (16 seconds) - Full arrangement, maximum energy
        print("Creating chorus...")
        chorus_chords = ['F', 'C', 'G', 'Am']
        chorus_audio = self.create_chorus(chorus_chords, 16.0)
        sections.append(('Chorus', chorus_audio))
        
        # 5. Verse 2 (16 seconds) - Variation of verse 1
        print("Creating verse 2...")
        verse2_audio = self.create_verse(verse_chords, 16.0, energy_level=0.7, variation=True)
        sections.append(('Verse 2', verse2_audio))
        
        # 6. Chorus 2 (16 seconds) - Full energy again
        print("Creating chorus 2...")
        chorus2_audio = self.create_chorus(chorus_chords, 16.0, variation=True)
        sections.append(('Chorus 2', chorus2_audio))
        
        # 7. Bridge (12 seconds) - Different chord progression, breakdown
        print("Creating bridge...")
        bridge_chords = ['Am', 'F', 'C', 'G']
        bridge_audio = self.create_bridge(bridge_chords, 12.0)
        sections.append(('Bridge', bridge_audio))
        
        # 8. Final Chorus (16 seconds) - Maximum intensity
        print("Creating final chorus...")
        final_chorus_audio = self.create_chorus(chorus_chords, 16.0, final=True)
        sections.append(('Final Chorus', final_chorus_audio))
        
        # 9. Outro (12 seconds) - Wind down
        print("Creating outro...")
        outro_audio = self.create_outro(['F', 'C'], 12.0)
        sections.append(('Outro', outro_audio))
        
        # Combine all sections
        full_track = np.concatenate([audio for _, audio in sections])
        
        # Final mastering
        print("Applying final mastering...")
        mastered_track = self.master_track(full_track)
        
        return mastered_track, sections
    
    def create_intro(self, chords, duration):
        """Create atmospheric intro with pad and filtered percussion"""
        # Pad sound
        pad_audio = self.create_atmospheric_pad(chords, duration)
        
        # Subtle percussion (filtered)
        drum_pattern = {
            'kick': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'snare': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'hihat_closed': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        drums_audio = self.drum_machine.sequence_pattern(drum_pattern)
        drums_audio = self.synthesizer.simple_lowpass_filter(drums_audio, 400, 1.0) * 0.4
        
        # Stretch/repeat to match duration
        target_samples = int(duration * self.sample_rate)
        if len(drums_audio) < target_samples:
            repeats = int(np.ceil(target_samples / len(drums_audio)))
            drums_audio = np.tile(drums_audio, repeats)[:target_samples]
        else:
            drums_audio = drums_audio[:target_samples]
        
        if len(pad_audio) < target_samples:
            pad_audio = np.pad(pad_audio, (0, target_samples - len(pad_audio)))
        else:
            pad_audio = pad_audio[:target_samples]
        
        # Mix and apply effects
        mixed = pad_audio * 0.8 + drums_audio
        mixed = self.effects.apply_reverb(mixed, room_size=0.8, damping=0.5)
        
        return mixed * 0.6  # Intro should be quieter
    
    def create_verse(self, chords, duration, energy_level=0.7, variation=False):
        """Create verse section with bass, drums, and subtle lead"""
        # Bass line
        bass_audio = self.create_synthwave_bass_line(chords, duration)
        
        # Drum pattern
        if variation:
            drum_pattern = {
                'kick': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                'snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat_closed': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                'hihat_open': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        else:
            drum_pattern = {
                'kick': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                'snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat_closed': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        
        drums_audio = self.drum_machine.sequence_pattern(drum_pattern)
        
        # Subtle lead melody
        lead_audio = self.create_subtle_lead(chords, duration)
        
        # Stretch to match duration
        target_samples = int(duration * self.sample_rate)
        drums_audio = self.match_length(drums_audio, target_samples)
        bass_audio = self.match_length(bass_audio, target_samples)
        lead_audio = self.match_length(lead_audio, target_samples)
        
        # Mix
        mixed = (drums_audio * 0.8 + 
                bass_audio * 0.7 + 
                lead_audio * 0.5)
        
        # Apply effects
        mixed = self.effects.apply_compression(mixed, threshold=0.7, ratio=3.0)
        
        return mixed * energy_level
    
    def create_buildup(self, chords, duration):
        """Create tension-building section"""
        # Rising noise sweep
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        noise = np.random.normal(0, 0.1, len(t))
        
        # Rising filter sweep
        start_freq = 200
        end_freq = 8000
        cutoff_curve = start_freq * (end_freq / start_freq) ** (t / duration)
        
        swept_noise = np.zeros_like(noise)
        for i in range(len(noise)):
            # Simple high-pass effect
            if i > 0:
                swept_noise[i] = noise[i] - noise[i-1] * 0.9
        
        # Volume ramp
        volume_ramp = np.linspace(0.1, 0.8, len(t))
        swept_noise *= volume_ramp
        
        # Add drums with increasing intensity
        drum_pattern = {
            'kick': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'snare': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'hihat_closed': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'clap': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        drums_audio = self.drum_machine.sequence_pattern(drum_pattern)
        drums_audio = self.match_length(drums_audio, len(swept_noise))
        
        # Rising bass
        bass_freq = self.theory.midi_to_freq(self.theory.note_name_to_midi('C', octave=2))
        rising_bass = np.sin(2 * np.pi * bass_freq * t) * np.linspace(0.2, 0.8, len(t))
        
        # Combine
        buildup = swept_noise + drums_audio * 0.6 + rising_bass * 0.5
        
        return buildup * 0.9
    
    def create_chorus(self, chords, duration, variation=False, final=False):
        """Create full-energy chorus section"""
        # Full drum pattern
        drum_pattern = {
            'kick': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'hihat_closed': [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            'hihat_open': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'clap': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        }
        
        drums_audio = self.drum_machine.sequence_pattern(drum_pattern)
        
        # Powerful bass
        bass_audio = self.create_synthwave_bass_line(chords, duration, punchy=True)
        
        # Lead synth
        lead_audio = self.create_powerful_lead(chords, duration)
        
        # Pad for fullness
        pad_audio = self.create_atmospheric_pad(chords, duration, bright=True)
        
        # Match lengths
        target_samples = int(duration * self.sample_rate)
        drums_audio = self.match_length(drums_audio, target_samples)
        bass_audio = self.match_length(bass_audio, target_samples)
        lead_audio = self.match_length(lead_audio, target_samples)
        pad_audio = self.match_length(pad_audio, target_samples)
        
        # Mix with more intensity for final chorus
        intensity = 1.2 if final else 1.0
        mixed = (drums_audio * 0.9 * intensity + 
                bass_audio * 0.8 * intensity + 
                lead_audio * 0.7 * intensity + 
                pad_audio * 0.4)
        
        # Effects
        if final:
            mixed = self.effects.apply_soft_distortion(mixed, gain=1.5)
        
        mixed = self.effects.apply_compression(mixed, threshold=0.6, ratio=4.0)
        
        return mixed
    
    def create_bridge(self, chords, duration):
        """Create contrasting bridge section"""
        # Filtered drums
        drum_pattern = {
            'kick': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'snare': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'hihat_closed': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        drums_audio = self.drum_machine.sequence_pattern(drum_pattern)
        drums_audio = self.effects.simple_lowpass_filter(drums_audio, 800, 1.0)
        
        # Atmospheric elements
        pad_audio = self.create_atmospheric_pad(chords, duration, dark=True)
        
        # Subtle arpeggiated melody
        arp_audio = self.create_arpeggiated_melody(chords, duration)
        
        # Match lengths
        target_samples = int(duration * self.sample_rate)
        drums_audio = self.match_length(drums_audio, target_samples)
        pad_audio = self.match_length(pad_audio, target_samples)
        arp_audio = self.match_length(arp_audio, target_samples)
        
        # Mix
        mixed = (drums_audio * 0.6 + 
                pad_audio * 0.7 + 
                arp_audio * 0.5)
        
        # Add reverb for space
        mixed = self.effects.apply_reverb(mixed, room_size=0.9, damping=0.3)
        
        return mixed * 0.8
    
    def create_outro(self, chords, duration):
        """Create peaceful outro"""
        # Soft pad
        pad_audio = self.create_atmospheric_pad(chords, duration)
        
        # Minimal percussion
        drum_pattern = {
            'kick': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'snare': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'hihat_closed': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'hihat_open': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'clap': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        drums_audio = self.drum_machine.sequence_pattern(drum_pattern) * 0.3
        
        # Fade out
        target_samples = int(duration * self.sample_rate)
        drums_audio = self.match_length(drums_audio, target_samples)
        pad_audio = self.match_length(pad_audio, target_samples)
        
        fade_out = np.linspace(1.0, 0.0, target_samples)
        
        mixed = (pad_audio * 0.8 + drums_audio) * fade_out
        
        return mixed
    
    def create_synthwave_bass_line(self, chords, duration, punchy=False):
        """Create powerful synthwave bass line"""
        chord_duration = duration / len(chords)
        total_samples = int(duration * self.sample_rate)
        bass_audio = np.zeros(total_samples)
        
        for i, chord_name in enumerate(chords):
            start_sample = int(i * chord_duration * self.sample_rate)
            
            # Get root note
            root_note = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] == '#':
                root_note += '#'
            
            bass_freq = self.theory.midi_to_freq(
                self.theory.note_name_to_midi(root_note, octave=2)
            )
            
            if punchy:
                bass_tone = self.synthesizer.create_synthwave_bass(bass_freq, chord_duration, cutoff=1200, resonance=3.0)
            else:
                bass_tone = self.synthesizer.create_synthwave_bass(bass_freq, chord_duration)
            
            end_sample = min(start_sample + len(bass_tone), total_samples)
            bass_audio[start_sample:end_sample] = bass_tone[:end_sample - start_sample]
        
        return bass_audio
    
    def create_atmospheric_pad(self, chords, duration, bright=False, dark=False):
        """Create atmospheric pad sound"""
        chord_duration = duration / len(chords)
        pad_audio, _ = self.progression_gen.create_custom_progression(
            [(chord[0], 'minor' if 'm' in chord else 'major', 4) for chord in chords],
            chord_duration=chord_duration,
            waveform='sine'
        )
        
        # Apply effects based on mood
        if bright:
            pad_audio = self.effects.simple_highpass_filter(pad_audio, 200, 1.0)
        elif dark:
            pad_audio = self.effects.simple_lowpass_filter(pad_audio, 800, 1.5)
        
        return pad_audio
    
    def create_subtle_lead(self, chords, duration):
        """Create subtle lead melody"""
        return self.synthesizer.create_synthwave_lead(440, duration) * 0.6
    
    def create_powerful_lead(self, chords, duration):
        """Create powerful lead synth"""
        return self.synthesizer.create_synthwave_lead(880, duration) * 0.8
    
    def create_arpeggiated_melody(self, chords, duration):
        """Create arpeggiated melody"""
        return self.synthesizer.create_synthwave_lead(660, duration) * 0.4
    
    def match_length(self, audio, target_length):
        """Match audio length to target by repeating or trimming"""
        if len(audio) < target_length:
            repeats = int(np.ceil(target_length / len(audio)))
            audio = np.tile(audio, repeats)[:target_length]
        else:
            audio = audio[:target_length]
        return audio
    
    def master_track(self, audio):
        """Apply final mastering to the complete track"""
        # Compression
        audio = self.effects.apply_compression(audio, threshold=0.7, ratio=3.0)
        
        # EQ (simulate with filters)
        audio = self.effects.simple_highpass_filter(audio, 20, 1.0)  # Remove sub-bass
        audio = self.effects.simple_lowpass_filter(audio, 18000, 1.0)  # Remove ultra-high
        
        # Limiting (prevent clipping)
        peak_level = np.max(np.abs(audio))
        if peak_level > 0.95:
            audio = audio / peak_level * 0.95
        
        # Normalize to -3dB
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio

def main():
    """Create and play the complete professional demo track"""
    print("=== Complete Music Production Demo ===")
    print("This demo showcases all concepts from the music production series:")
    print("• Digital Audio • Synthesis • Effects • Drums • Harmony • Arrangement")
    print()
    
    # Create the music producer
    producer = CompleteMusicProducer(bpm=128, key='C')
    
    # Create the complete track
    track, sections = producer.create_professional_track()
    
    print(f"\nTrack created! Duration: {len(track) / producer.sample_rate:.1f} seconds")
    print("\nSection breakdown:")
    current_time = 0
    for name, audio in sections:
        section_duration = len(audio) / producer.sample_rate
        print(f"  {current_time:.1f}s - {current_time + section_duration:.1f}s: {name}")
        current_time += section_duration
    
    # Save the track
    sf.write("complete_professional_track.wav", track, producer.sample_rate)
    print(f"\nSaved as 'complete_professional_track.wav'")
    
    # Play the track
    print("\nPlaying complete track...")
    sd.play(track, producer.sample_rate)
    sd.wait()
    
    # Visualize the track
    print("\nGenerating track visualization...")
    
    plt.figure(figsize=(16, 10))
    
    # Waveform
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(track) / producer.sample_rate, len(track))
    plt.plot(time_axis, track)
    plt.title('Complete Professional Track - Waveform')
    plt.ylabel('Amplitude')
    
    # Add section markers
    current_time = 0
    for name, audio in sections:
        section_duration = len(audio) / producer.sample_rate
        plt.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
        plt.text(current_time + section_duration/2, 0.8, name, 
                fontsize=9, ha='center', rotation=45)
        current_time += section_duration
    
    plt.grid(True, alpha=0.3)
    
    # Frequency spectrum
    plt.subplot(3, 1, 2)
    fft = np.fft.fft(track[::10])  # Downsample for faster processing
    frequencies = np.fft.fftfreq(len(fft), 10/producer.sample_rate)
    magnitude = np.abs(fft)
    
    positive_freq_mask = (frequencies > 0) & (frequencies < 10000)
    plt.semilogx(frequencies[positive_freq_mask], 20 * np.log10(magnitude[positive_freq_mask] + 1e-10))
    plt.title('Frequency Spectrum (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    
    # Spectrogram
    plt.subplot(3, 1, 3)
    # Downsample for visualization
    downsampled = track[::100]
    sample_rate_down = producer.sample_rate // 100
    
    plt.specgram(downsampled, Fs=sample_rate_down, cmap='viridis')
    plt.title('Spectrogram - Frequency Content Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power (dB)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Complete Music Production Concepts Demonstrated ===")
    print("✓ Digital Audio: Sample rates, bit depth, waveforms")
    print("✓ Synthesis: Oscillators, ADSR envelopes, LFOs, FM synthesis")
    print("✓ Effects: Reverb, delay, compression, distortion, filtering")
    print("✓ Drums: Synthesized percussion, patterns, velocity layers")
    print("✓ Music Theory: Scales, chords, progressions, harmonic rhythm")
    print("✓ Arrangement: Song structure, dynamics, instrumentation")
    print("✓ Mixing: Balance, EQ, compression, stereo imaging")
    print("✓ Mastering: Final polish, limiting, normalization")
    print()
    print("This demonstrates a complete electronic music production workflow!")
    print("From basic digital audio concepts to a finished, professional track.")

if __name__ == "__main__":
    main() 