#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from speechbrain.inference import EncoderClassifier
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class XVectorExtractor:
    def __init__(self):
        # Load the pre-trained x-vector model from SpeechBrain
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        )

    def extract(self, audio_path):
        # Load the audio file
        signal, fs = torchaudio.load(audio_path)

        # Ensure the audio is mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # Extract the x-vector embedding
        embeddings = self.model.encode_batch(signal)
        return embeddings.squeeze().detach().numpy()

def load_audio(file_path, sr=16000):
    """Load an audio file."""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def extract_mfcc(audio, sr, n_mfcc=13):
    """Extract MFCC features from the audio signal."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def signal_processing_anonymization(audio, sr, pitch_shift_steps=2):
    """
    Apply classical signal processing modifications:
      - Pitch shifting to modify spectral properties.
      - (Optional) formant shifting or time-stretching can be added.
    """
    # Pitch shift: increasing pitch by a fixed number of semitones
    audio_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_steps)
    return audio_shifted

def anonymize_xvector(xvector, noise_level=0.1):
    """
    Anonymize the x-vector by adding a controlled amount of noise.
    In practice, you might apply a learned transformation or replace the vector
    with one sampled from a pool of non-identifiable speakers.
    """
    # Add random noise to simulate anonymization
    anonymized_vector = xvector + noise_level * np.random.randn(*xvector.shape)
    return anonymized_vector

def synthesize_speech(mfcc, anonymized_xvector):
    """
    Synthesize the final speech signal.
    A neural vocoder would take spectral features (MFCC or mel-spectrogram) along with
    the anonymized speaker embedding to generate speech.
    Here we simply use an inverse transformation as a placeholder.
    """
    # This is a placeholder synthesis. In practice, use a neural vocoder like WaveGlow.
    # For instance, you could combine the MFCC with the x-vector to condition a TTS model.
    synthesized_audio = librosa.feature.inverse.mfcc_to_audio(mfcc)
    return synthesized_audio


def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`, 
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """

    # Apply your anonymization algorithm

    # Step 1: Load the audio clip
    audio, sr = load_audio(input_audio_path)
    print("Audio loaded, duration:", len(audio)/sr, "seconds")

    # Step 2: Extract low-level features (e.g., MFCCs)
    mfcc_features = extract_mfcc(audio, sr)
    print("Extracted MFCC features with shape:", mfcc_features.shape)


    # Step 3: Apply signal processing-based anonymization (e.g., pitch shifting)
    processed_audio = signal_processing_anonymization(audio, sr, pitch_shift_steps=2)
    print("Applied pitch shift for basic anonymization.")

     # Step 4:Initialize the x-vector extractor & Extract the x-vector
    extractor = XVectorExtractor()
    xvector = extractor.extract(input_audio_path)
    print("Extracted x-vector of shape:", xvector.shape)

    # Step 5: Anonymize the x-vector (simulate domain adaptation and transformation)
    anonymized_xvector = anonymize_xvector(xvector, noise_level=0.1)
    print("Anonymized the x-vector.")


    # Step 6: Synthesize the final anonymized speech by combining both solutions
    final_audio = synthesize_speech(mfcc_features, xvector)


    # Output:
    audio = final_audio
    sr = sr
    
    return audio, sr

