#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

from amine_model import cleverlytics_anonymization_algorithm

def anonymize(input_audio_path, params): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
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

    # mcadams_coeff=0.9
    # pitch_shift_steps=3
    # gain_db=15
    # use_noise_reduction=True
    # mfcc_encryption=True

    # params = {'mcadams_coeff': mcadams_coeff, 'pitch_shift_steps': pitch_shift_steps, 'gain_db': gain_db, 'use_noise_reduction': use_noise_reduction, 'mfcc_encryption': mfcc_encryption}

    audio, sr = cleverlytics_anonymization_algorithm(input_audio_path, **params)

    
    return audio, sr