import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import hashlib
from tqdm import tqdm
import argparse

def cleverlytics_anonymization_algorithm(
    input_directory, 
    output_directory, 
    mcadams_coeff=1, 
    pitch_shift_steps=3, 
    gain_db=15, 
    use_noise_reduction=False, 
    mfcc_encryption=True
):
    """
    Anonymize multiple audio files in a directory using multiple techniques.
    
    Args:
        input_directory (str): Path to directory containing input audio files
        output_directory (str): Path to directory for saving anonymized files
        mcadams_coeff (float, optional): McAdams coefficient for spectral envelope. Defaults to 1.
        pitch_shift_steps (int, optional): Number of semitones to shift pitch. Defaults to 3.
        gain_db (float, optional): Volume increase in decibels. Defaults to 15.
        use_noise_reduction (bool, optional): Whether to apply noise reduction. Defaults to False.
        mfcc_encryption (bool, optional): Whether to apply MFCC hashing. Defaults to True.
    """
    os.makedirs(output_directory, exist_ok=True)

    def apply_mcadams_coefficient(frequency, mcadams_coeff):
        f0 = 0  # Reference frequency
        return f0 + (frequency - f0) ** mcadams_coeff

    def anonymize_spectral_envelope(spectral_envelope, mcadams_coeff):
        n_freqs = len(spectral_envelope)
        freq_bins = np.linspace(0, 1, n_freqs)
        transformed_freqs = apply_mcadams_coefficient(freq_bins, mcadams_coeff)
        return np.interp(transformed_freqs, freq_bins, spectral_envelope)

    def process_audio_file(input_path):
        y, sr = librosa.load(input_path)
        stft = librosa.stft(y, n_fft=1024, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        transformed_magnitude = np.copy(magnitude)
        for i in range(magnitude.shape[1]):
            transformed_magnitude[:, i] = anonymize_spectral_envelope(magnitude[:, i], mcadams_coeff)
        transformed_stft = transformed_magnitude * np.exp(1j * phase)
        y1 = librosa.istft(transformed_stft, hop_length=512)
        if mfcc_encryption:
            mfccs = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
            hashed_mfccs = np.zeros_like(mfccs)
            for i in range(mfccs.shape[1]):
                for j in range(mfccs.shape[0]):
                    feature = mfccs[j, i]
                    hashed_feature = hashlib.sha256(str(feature).encode()).hexdigest()
                    hashed_mfccs[j, i] = int(hashed_feature[:8], 16) / (2**32 - 1)
            mel_spec = librosa.feature.melspectrogram(y=y1, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            mel_spec[:13, :] = hashed_mfccs
            stft = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)
            y2 = librosa.griffinlim(stft, hop_length=512, win_length=2048)
        else:
            y2 = y1
        y3 = librosa.effects.pitch_shift(y2, sr=sr, n_steps=pitch_shift_steps)
        y3 = y3.speedup(playback_speed=1.1)
        y4 = nr.reduce_noise(y=y3, sr=sr) if use_noise_reduction else y3
        gain = 10 ** (gain_db / 20)
        y5 = np.clip(y4 * gain, -1.0, 1.0)
        return y5, sr

    audio_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    def create_filename(base_name, params):
        return f"{base_name}_MC{int(params['mcadams_coeff']*10)}_PS{params['pitch_shift_steps']}_G{params['gain_db']}_NR{'T' if params['use_noise_reduction'] else 'F'}_MCENC{'T' if params['mfcc_encryption'] else 'F'}.wav"

    for filename in tqdm(audio_files, desc="Anonymizing Audio Files"):
        input_path = os.path.join(input_directory, filename)
        params = {'mcadams_coeff': mcadams_coeff, 'pitch_shift_steps': pitch_shift_steps, 'gain_db': gain_db, 'use_noise_reduction': use_noise_reduction, 'mfcc_encryption': mfcc_encryption}
        output_filename = create_filename(filename, params)
        output_path = os.path.join(output_directory, output_filename)
        try:
            anonymized_audio, sr = process_audio_file(input_path)
            sf.write(output_path, anonymized_audio, sr)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Anonymization procedure is complete!")

def main():
    parser = argparse.ArgumentParser(description='Anonymize audio files in a directory.')
    parser.add_argument('input_directory', help='Path to input directory containing audio files')
    parser.add_argument('output_directory', help='Path to output directory for anonymized files')
    parser.add_argument('--mcadams_coeff', type=float, default=1, help='McAdams coefficient for spectral envelope (default: 1)')
    parser.add_argument('--pitch_shift_steps', type=int, default=3, help='Number of semitones to shift pitch (default: 3)')
    parser.add_argument('--gain_db', type=float, default=15, help='Volume increase in decibels (default: 15)')
    parser.add_argument('--no_noise_reduction', action='store_true', help='Disable noise reduction (default: False)')
    parser.add_argument('--no_mfcc_encryption', action='store_true', help='Disable MFCC encryption (default: enabled)')
    args = parser.parse_args()
    cleverlytics_anonymization_algorithm(
        args.input_directory,
        args.output_directory,
        mcadams_coeff=args.mcadams_coeff,
        pitch_shift_steps=args.pitch_shift_steps,
        gain_db=args.gain_db,
        use_noise_reduction=not args.no_noise_reduction,
        mfcc_encryption=not args.no_mfcc_encryption
    )

if __name__ == '__main__':
    main()