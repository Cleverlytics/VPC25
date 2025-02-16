import os
import numpy as np
import librosa
from scipy.signal import butter, filtfilt

#############################################################################
# YOUR ANONYMIZATION MODEL
#############################################################################
def anonymize(input_audio_path):
    """
    Anonymization algorithm.
    
    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in ".wav" format.
    
    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type np.float32.
    sr : int
        The sample rate of the processed audio.
    """
    # Étape 1 : Charger le fichier audio
    audio, sr = librosa.load(input_audio_path, sr=None)
    
    # Étape 2 : Prétraitement (normalisation, suppression de bruit, filtrage passe-bas)
    audio = normalize_amplitude(audio)
    audio = remove_noise(audio, sr)
    audio = apply_low_pass_filter(audio, sr)
    
    # Étape 3 : Appliquer les techniques d'anonymisation
    audio = controlled_randomization(audio)  # Randomisation Contrôlée
    audio = advanced_pseudonymization(audio, sr)  # Pseudonymisation Avancée
    
    # Étape 4 : Post-traitement (assurer la compatibilité avec soundfile.write)
    audio = audio.astype(np.float32)
    
    return audio, sr

#############################################################################
# Fonctions Utilitaires
#############################################################################

# Normalisation de l'amplitude
def normalize_amplitude(audio):
    return audio / np.max(np.abs(audio))

# Suppression de bruit
def remove_noise(audio, sr):
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr)

# Filtre passe-bas
def apply_low_pass_filter(audio, sr, cutoff_freq=4000):
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, audio)

# Randomisation Contrôlée
def controlled_randomization(audio, noise_level=0.001):
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

# Pseudonymisation Avancée
def advanced_pseudonymization(audio, sr, mfcc_noise_level=0.02):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    modified_mfccs = mfccs + np.random.normal(0, mfcc_noise_level, mfccs.shape)
    return librosa.feature.inverse.mfcc_to_audio(modified_mfccs, sr=sr)