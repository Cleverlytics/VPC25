
    #############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################



# Fonction pour charger et inspecter les fichiers audio
def load_and_inspect_audio(file_path):
    # Initialisation des listes pour stocker les métadonnées
    file_paths = []
    sampling_rates = []
    durations = []
    is_noisy = []

    # Charger le fichier audio
    audio, sr = librosa.load(file_path, sr=None)
    sampling_rates.append(sr)
    durations.append(len(audio) / sr)

    # Identifier les fichiers bruités (simple heuristique basée sur l'énergie)
    energy = np.sum(audio**2) / len(audio)
    is_noisy.append("Yes" if energy < 0.001 else "No")  # Seuil arbitraire pour détecter le bruit

    # Créer un DataFrame pour organiser les métadonnées
    metadata = pd.DataFrame({
        "File Path": file_paths,
        "Sampling Rate (Hz)": sampling_rates,
        "Duration (s)": durations,
        "Is Noisy": is_noisy
    })

    return metadata

# Exécution de la fonction
metadata = load_and_inspect_audio(data_path)

def normalize_sampling_rate(file_path, target_sr=16000):
    # Charger le fichier audio
    audio, sr = librosa.load(file_path, sr=None)
    # Rééchantillonner à la fréquence cible
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_resampled, target_sr

# Appliquer la normalisation à tous les fichiers
metadata["Audio"], metadata["Normalized Sampling Rate"] = zip(
    *metadata["File Path"].apply(lambda x: normalize_sampling_rate(x))
)

def calculate_snr(audio):
    # Calculer le rapport signal/bruit (SNR)
    signal_power = np.sum(audio**2)
    noise_power = np.sum((audio - np.mean(audio))**2)
    return 10 * np.log10(signal_power / noise_power)

# Ajouter une colonne SNR au DataFrame
metadata["SNR"] = metadata["Audio"].apply(calculate_snr)

# Mettre à jour la détection de fichiers bruités
metadata["Is Noisy"] = metadata["SNR"].apply(lambda x: "Yes" if x < 20 else "No")  # Seuil SNR ajusté

# Fonction pour normaliser l'amplitude
def normalize_amplitude(audio):
    # Normaliser l'amplitude entre -1 et 1
    return audio / np.max(np.abs(audio))

# Fonction pour supprimer le bruit
def remove_noise(audio, sr):
    # Appliquer la suppression de bruit avec Noisereduce
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise

# Fonction pour appliquer un filtre passe-bas
def apply_low_pass_filter(audio, sr, cutoff_freq=4000):
    # Conception du filtre passe-bas
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

# Prétraitement des fichiers audio
def preprocess_audio(metadata):
    preprocessed_audio = []
    for idx, row in metadata.iterrows():
        audio = row["Audio"]
        sr = row["Normalized Sampling Rate"]

        # Étape 1 : Normalisation de l'amplitude
        normalized_audio = normalize_amplitude(audio)

        # Étape 2 : Suppression du bruit
        if row["Is Noisy"] == "Yes":
            denoised_audio = remove_noise(normalized_audio, sr)
        else:
            denoised_audio = normalized_audio

        # Étape 3 : Application d'un filtre passe-bas (optionnel)
        filtered_audio = apply_low_pass_filter(denoised_audio, sr)

        # Stocker le fichier prétraité
        preprocessed_audio.append(filtered_audio)

    # Ajouter les fichiers prétraités au DataFrame
    metadata["Preprocessed Audio"] = preprocessed_audio
    return metadata

# Exécution du prétraitement
metadata = preprocess_audio(metadata)

# Visualisation d'un exemple de fichier prétraité
example_file = metadata.iloc[0]["File Path"]
preprocessed_audio = metadata.iloc[0]["Preprocessed Audio"]
sr = metadata.iloc[0]["Normalized Sampling Rate"]

# Fonction pour la Modification Spectrale
def spectral_modification(audio, sr):
    # Calcul de la Transformée de Fourier
    fft_audio = fft.fft(audio)
    frequencies = np.fft.fftfreq(len(fft_audio), 1 / sr)

    # Altérer les formants (ex. : atténuer les fréquences entre 500 Hz et 2000 Hz)
    for i, freq in enumerate(frequencies):
        if 500 <= abs(freq) <= 2000:  # Bande critique pour les formants
            fft_audio[i] *= 0.5  # Réduire l'amplitude de ces fréquences

    # Inverse FFT pour reconstruire le signal
    modified_audio = np.real(fft.ifft(fft_audio))
    return modified_audio

# Fonction pour la Randomisation Contrôlée
def controlled_randomization(audio):
    # Ajouter un bruit gaussien minimal
    noise = np.random.normal(0, 0.01, len(audio))  # Bruit gaussien avec une faible amplitude
    randomized_audio = audio + noise
    return randomized_audio

# Fonction pour la Pseudonymisation Avancée
def advanced_pseudonymization(audio, sr):
    # Extraire les caractéristiques vocales (MFCC)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Modifier légèrement les MFCC pour créer une "pseudo-voix"
    modified_mfccs = mfccs + np.random.normal(0, 0.1, mfccs.shape)

    # Reconstruire le signal à partir des MFCC modifiés
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(modified_mfccs, sr=sr)
    return reconstructed_audio

# Appliquer les techniques individuellement
def apply_individual_techniques(metadata):
    results = []
    for idx, row in metadata.iterrows():
        audio = row["Preprocessed Audio"]
        sr = row["Normalized Sampling Rate"]

        # Technique 1 : Modification Spectrale
        spectral_modified = spectral_modification(audio, sr)

        # Technique 2 : Randomisation Contrôlée
        randomized = controlled_randomization(audio)

        # Technique 3 : Pseudonymisation Avancée
        pseudonymized = advanced_pseudonymization(audio, sr)

        # Stocker les résultats
        results.append({
            "File Path": row["File Path"],
            "Spectral Modified": spectral_modified,
            "Randomized": randomized,
            "Pseudonymized": pseudonymized
        })

    return pd.DataFrame(results)

# Exécution des techniques individuelles
results = apply_individual_techniques(metadata)

# Fonction pour calculer le WER (Word Error Rate)
def calculate_wer(original_text, anonymized_text):
    # Tokeniser les phrases
    original_words = original_text.split()
    anonymized_words = anonymized_text.split()

    # Validation : Vérifier si les textes sont non vides
    if len(original_words) == 0 or len(anonymized_words) == 0:
        print("Erreur : Texte original ou anonymisé vide. Impossible de calculer le WER.")
        return 1.0  # Retourner une valeur par défaut en cas d'erreur

    # Calculer le nombre d'erreurs
    errors = sum(1 for o, a in zip(original_words, anonymized_words) if o != a)

    # Calculer le WER
    wer = errors / len(original_words)
    return wer

# Fonction pour calculer l'EER (Equal Error Rate)
def calculate_eer(embeddings_original, embeddings_anonymized):
    # Validation : S'assurer que les embeddings sont bien des tableaux NumPy
    if not isinstance(embeddings_original, np.ndarray) or not isinstance(embeddings_anonymized, np.ndarray):
        raise ValueError("Les embeddings doivent être des tableaux NumPy.")

    # Vérifier la forme des embeddings
    embeddings_original = embeddings_original.flatten()  # Assure un vecteur 1D
    embeddings_anonymized = embeddings_anonymized.flatten()
    if embeddings_original.ndim != 1 or embeddings_anonymized.ndim != 1:
        raise ValueError("Les embeddings doivent être des vecteurs 1D après redimensionnement.")
    # Calculer la similarité cosinus directement entre les deux vecteurs
    eer = cosine(embeddings_original, embeddings_anonymized)  # Distance cosinus unique
    return eer

# Fonction pour évaluer la naturalité (subjective)
def evaluate_naturalness(audio_files):
    # Simuler une évaluation subjective avec des scores aléatoires (à remplacer par des tests humains)
    naturalness_scores = []
    for audio in audio_files:
        score = np.random.uniform(3, 5)  # Score entre 3 et 5 (échelle de Likert)
        naturalness_scores.append(score)
    return naturalness_scores

# Transcription avec Whisper
def transcribe_audio_whisper(file_path):
    # Charger le modèle Whisper (base est un bon compromis entre performance et rapidité)
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]  # Retourne la transcription

# Évaluation des performances
def evaluate_techniques(results):
    performance_metrics = []
    for idx, row in results.iterrows():
        # Données originales
        original_audio = metadata.iloc[idx]["Preprocessed Audio"]
        sampling_rate = metadata.iloc[idx]["Normalized Sampling Rate"]  # Renommage de 'sr'

        # Techniques anonymisées
        spectral_modified = row["Spectral Modified"]
        randomized = row["Randomized"]
        pseudonymized = row["Pseudonymized"]

        # Exemple de texte original (pour WER)
        original_text = "This is a sample text for evaluation."  # Remplacer par le texte réel

        # Conversion audio → texte avec Whisper
        try:
            original_text_asr = transcribe_audio_whisper(row["File Path"])
        except Exception as e:
            print(f"Erreur lors de la transcription avec Whisper pour {row['File Path']} : {e}")
            original_text_asr = original_text  # Fallback en cas d'erreur

        # Simuler les transcriptions anonymisées
        anonymized_text_spectral = "This is a modified spectral text."
        anonymized_text_randomized = "This is a randomized text."
        anonymized_text_pseudonymized = "This is a pseudonymized text."

        # Calcul du WER pour chaque technique
        wer_spectral = calculate_wer(original_text, anonymized_text_spectral)
        wer_randomized = calculate_wer(original_text, anonymized_text_randomized)
        wer_pseudonymized = calculate_wer(original_text, anonymized_text_pseudonymized)

        # Calcul de l'EER (approximation simplifiée)
        embeddings_original = np.random.rand(13)  # Remplacer par des embeddings réels
        embeddings_spectral = np.random.rand(13)
        embeddings_randomized = np.random.rand(13)
        embeddings_pseudonymized = np.random.rand(13)

        # Validation et redimensionnement des embeddings
        embeddings_original = np.array(embeddings_original).flatten()  # Assure un vecteur 1D
        embeddings_spectral = np.array(embeddings_spectral).flatten()
        embeddings_randomized = np.array(embeddings_randomized).flatten()
        embeddings_pseudonymized = np.array(embeddings_pseudonymized).flatten()

        # Vérification des dimensions
        if embeddings_original.ndim != 1 or embeddings_spectral.ndim != 1:
            raise ValueError("Les embeddings ne sont pas des vecteurs 1D après redimensionnement.")

        eer_spectral = calculate_eer(np.array(embeddings_original), np.array(embeddings_spectral))
        eer_randomized = calculate_eer(np.array(embeddings_original), np.array(embeddings_randomized))
        eer_pseudonymized = calculate_eer(np.array(embeddings_original), np.array(embeddings_pseudonymized))

        # Évaluation de la naturalité
        naturalness_scores = evaluate_naturalness([spectral_modified, randomized, pseudonymized])

        # Stocker les résultats
        performance_metrics.append({
            "File Path": row["File Path"],
            "WER Spectral": wer_spectral,
            "WER Randomized": wer_randomized,
            "WER Pseudonymized": wer_pseudonymized,
            "EER Spectral": eer_spectral,
            "EER Randomized": eer_randomized,
            "EER Pseudonymized": eer_pseudonymized,
            "Naturalness Spectral": naturalness_scores[0],
            "Naturalness Randomized": naturalness_scores[1],
            "Naturalness Pseudonymized": naturalness_scores[2]
        })
    return pd.DataFrame(performance_metrics)

# Exécution de l'évaluation
performance_results = evaluate_techniques(results)

# Fonction pour comparer les transcriptions
def compare_transcriptions(original_texts, whisper_texts):
    """
    Compare les transcriptions originales (humaines) avec celles générées par Whisper.
    Retourne le taux d'exactitude global.
    """
    accuracy = []
    for original, whisper in zip(original_texts, whisper_texts):
        original_words = original.split()
        whisper_words = whisper.split()
        correct = sum(1 for o, w in zip(original_words, whisper_words) if o == w)
        accuracy.append(correct / len(original_words))
    return np.mean(accuracy)

# Fonction pour visualiser les métriques
def visualize_metrics(performance_results):
    """
    Visualise les métriques WER, EER, et Naturalité pour chaque technique.
    """
    techniques = ["Spectral", "Randomized", "Pseudonymized"]
    metrics = ["WER", "EER", "Naturalness"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))

    for i, metric in enumerate(metrics):
        values = [performance_results[f"{metric} {tech}"].mean() for tech in techniques]
        axes[i].bar(techniques, values, color=['blue', 'green', 'orange'])
        axes[i].set_title(f"Comparison of {metric}")
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    plt.show()

# Tests subjectifs (simulés)
def subjective_tests(audio_files):
    """
    Simule des tests subjectifs pour évaluer la naturalité et l'intelligibilité.
    """
    scores = {
        "Naturalness": [],
        "Intelligibility": []
    }
    for audio in audio_files:
        naturalness_score = np.random.uniform(3, 5)  # Score entre 3 et 5 (échelle de Likert)
        intelligibility_score = np.random.uniform(3, 5)  # Score entre 3 et 5 (échelle de Likert)
        scores["Naturalness"].append(naturalness_score)
        scores["Intelligibility"].append(intelligibility_score)
    return scores

audio_files = performance_results["File Path"].tolist()
subjective_scores = subjective_tests(audio_files)

# Ajout des scores subjectifs au DataFrame
performance_results["Subjective Naturalness"] = subjective_scores["Naturalness"]
performance_results["Subjective Intelligibility"] = subjective_scores["Intelligibility"]

# Fonction pour la Randomisation Contrôlée
def controlled_randomization(audio):
    # Ajouter un bruit gaussien minimal
    noise = np.random.normal(0, 0.01, len(audio))  # Bruit gaussien avec une faible amplitude
    randomized_audio = audio + noise
    return randomized_audio

# Fonction pour la Pseudonymisation Avancée
def advanced_pseudonymization(audio, sr):
    # Extraire les caractéristiques vocales (MFCC)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Modifier légèrement les MFCC pour créer une "pseudo-voix"
    modified_mfccs = mfccs + np.random.normal(0, 0.1, mfccs.shape)

    # Reconstruire le signal à partir des MFCC modifiés
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(modified_mfccs, sr=sr)
    return reconstructed_audio

# Fusion des techniques
def combine_techniques(metadata):
    combined_results = []
    for idx, row in metadata.iterrows():
        audio = row["Preprocessed Audio"]
        sr = row["Normalized Sampling Rate"]

        # Étape 1 : Randomisation Contrôlée
        randomized_audio = controlled_randomization(audio)

        # Étape 2 : Pseudonymisation Avancée
        pseudonymized_audio = advanced_pseudonymization(randomized_audio, sr)

        # Stocker le résultat combiné
        combined_results.append({
            "File Path": row["File Path"],
            "Combined Audio": pseudonymized_audio
        })

    return pd.DataFrame(combined_results)

# Exécution de la fusion
combined_results = combine_techniques(metadata)

# Fonction pour calculer le WER (Word Error Rate)
def calculate_wer(original_text, anonymized_text):
    # Tokeniser les phrases
    original_words = original_text.split()
    anonymized_words = anonymized_text.split()
    # Calculer le nombre d'erreurs
    errors = sum(1 for o, a in zip(original_words, anonymized_words) if o != a)
    wer = errors / len(original_words)
    return wer

# Fonction pour calculer l'EER (Equal Error Rate)
def calculate_eer(embeddings_original, embeddings_anonymized):
    # Validation : S'assurer que les embeddings sont bien des tableaux NumPy
    if not isinstance(embeddings_original, np.ndarray) or not isinstance(embeddings_anonymized, np.ndarray):
        raise ValueError("Les embeddings doivent être des tableaux NumPy.")

    # Vérifier la forme des embeddings
    embeddings_original = embeddings_original.flatten()  # Assure un vecteur 1D
    embeddings_anonymized = embeddings_anonymized.flatten()
    if embeddings_original.ndim != 1 or embeddings_anonymized.ndim != 1:
        raise ValueError("Les embeddings doivent être des vecteurs 1D après redimensionnement.")
    # Calculer la similarité cosinus directement entre les deux vecteurs
    eer = cosine(embeddings_original, embeddings_anonymized)  # Distance cosinus unique
    return eer

# Transcription avec Whisper
def transcribe_audio_whisper(file_path):
    # Charger le modèle Whisper (base est un bon compromis entre performance et rapidité)
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]  # Retourne la transcription

# Évaluation des performances
def evaluate_combined_techniques(combined_results):
    performance_metrics = []
    for idx, row in combined_results.iterrows():
        # Données originales
        file_path = row["File Path"]
        combined_audio = row["Combined Audio"]

        # Exemple de texte original (pour WER)
        original_text = "This is a sample text for evaluation."  # Remplacer par le texte réel

        # Conversion audio → texte avec Whisper
        try:
            original_text_asr = transcribe_audio_whisper(file_path)
        except Exception as e:
            print(f"Erreur lors de la transcription avec Whisper pour {file_path} : {e}")
            original_text_asr = original_text  # Fallback en cas d'erreur

        # Simuler les transcriptions anonymisées
        anonymized_text_combined = "This is a combined anonymized text."

        # Calcul du WER pour la solution combinée
        wer_combined = calculate_wer(original_text, anonymized_text_combined)

        # Calcul de l'EER (approximation simplifiée)
        embeddings_original = np.random.rand(13)  # Remplacer par des embeddings réels
        embeddings_combined = np.random.rand(13)

        # Validation et redimensionnement des embeddings
        embeddings_original = np.array(embeddings_original).flatten()  # Assure un vecteur 1D
        embeddings_combined = np.array(embeddings_combined).flatten()

        # Vérification des dimensions
        if embeddings_original.ndim != 1 or embeddings_combined.ndim != 1:
            raise ValueError("Les embeddings ne sont pas des vecteurs 1D après redimensionnement.")

        eer_combined = calculate_eer(np.array(embeddings_original), np.array(embeddings_combined))

        # Stocker les résultats
        performance_metrics.append({
            "File Path": file_path,
            "WER Combined": wer_combined,
            "EER Combined": eer_combined
        })
    return pd.DataFrame(performance_metrics)

# Exécution de l'évaluation
final_results = evaluate_combined_techniques(combined_results)

# Fonction pour la Randomisation Contrôlée (réduction de l'amplitude)
def controlled_randomization(audio, noise_level=0.005):
    # Ajouter un bruit gaussien minimal
    noise = np.random.normal(0, noise_level, len(audio))  # Bruit gaussien avec une faible amplitude
    randomized_audio = audio + noise
    return randomized_audio

# Fonction pour la Pseudonymisation Avancée (réduction des modifications MFCC)
def advanced_pseudonymization(audio, sr, mfcc_noise_level=0.05):
    # Extraire les caractéristiques vocales (MFCC)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Modifier légèrement les MFCC pour créer une "pseudo-voix"
    modified_mfccs = mfccs + np.random.normal(0, mfcc_noise_level, mfccs.shape)

    # Reconstruire le signal à partir des MFCC modifiés
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(modified_mfccs, sr=sr)
    return reconstructed_audio

# Fusion des techniques avec ajustements
def combine_techniques(metadata):
    combined_results = []
    for idx, row in metadata.iterrows():
        audio = row["Preprocessed Audio"]
        sr = row["Normalized Sampling Rate"]

        # Étape 1 : Randomisation Contrôlée (réduction de l'amplitude)
        randomized_audio = controlled_randomization(audio, noise_level=0.005)

        # Étape 2 : Pseudonymisation Avancée (réduction des modifications MFCC)
        pseudonymized_audio = advanced_pseudonymization(randomized_audio, sr, mfcc_noise_level=0.05)

        # Stocker le résultat combiné
        combined_results.append({
            "File Path": row["File Path"],
            "Combined Audio": pseudonymized_audio
        })

    return pd.DataFrame(combined_results)

# Exécution de la fusion
combined_results = combine_techniques(metadata)

# Fonction pour calculer le WER (Word Error Rate)
def calculate_wer(original_text, anonymized_text):
    # Tokeniser les phrases
    original_words = original_text.split()
    anonymized_words = anonymized_text.split()

    # Validation : Vérifier si les textes sont non vides
    if len(original_words) == 0 or len(anonymized_words) == 0:
        print("Erreur : Texte original ou anonymisé vide. Impossible de calculer le WER.")
        return 1.0  # Retourner une valeur par défaut en cas d'erreur

    # Calculer le nombre d'erreurs
    errors = sum(1 for o, a in zip(original_words, anonymized_words) if o != a)

    # Calculer le WER
    wer = errors / len(original_words)
    return wer

# Évaluation des performances avec validation des transcriptions
def evaluate_combined_techniques(combined_results):
    performance_metrics = []
    for idx, row in combined_results.iterrows():
        # Données originales
        file_path = row["File Path"]
        combined_audio = row["Combined Audio"]
        sr = metadata.iloc[idx]["Normalized Sampling Rate"]

        # Prétraitement pour Whisper
        preprocessed_audio = preprocess_for_whisper(combined_audio, sr)

        # Sauvegarder l'audio prétraité temporairement
        temp_file = "temp_preprocessed_audio.wav"
        wavfile.write(temp_file, sr, preprocessed_audio)

        # Conversion audio → texte avec Whisper
        try:
            original_text_asr = transcribe_audio_whisper(temp_file)
        except Exception as e:
            print(f"Erreur lors de la transcription avec Whisper pour {file_path} : {e}")
            original_text_asr = "This is a sample text for evaluation."  # Fallback

        # Simuler les transcriptions anonymisées
        anonymized_text_combined = "This is a combined anonymized text."

        # Validation des transcriptions
        if not original_text_asr.strip() or not anonymized_text_combined.strip():
            print(f"Transcription vide détectée pour {file_path}. Utilisation de valeurs par défaut.")
            wer_combined = 1.0  # Valeur par défaut en cas de texte vide
            eer_combined = 1.0  # Valeur par défaut en cas de texte vide
        else:
            # Calcul du WER pour la solution combinée
            wer_combined = calculate_wer(original_text_asr, anonymized_text_combined)

            # Calcul de l'EER (approximation simplifiée)
            embeddings_original = np.random.rand(13)  # Remplacer par des embeddings réels
            embeddings_combined = np.random.rand(13)
            eer_combined = calculate_eer(embeddings_original, embeddings_combined)

        # Stocker les résultats
        performance_metrics.append({
            "File Path": file_path,
            "WER Combined": wer_combined,
            "EER Combined": eer_combined
        })
    return pd.DataFrame(performance_metrics)

# Exécution de l'évaluation
final_results = evaluate_combined_techniques(combined_results)

# Comparaison des modèles Whisper
models = ["base", "medium", "large"]
results_by_model = {}

for model_name in models:
    print(f"Évaluation avec le modèle Whisper : {model_name}")
    whisper_model = whisper.load_model(model_name)

    def transcribe_with_model(file_path):
        result = whisper_model.transcribe(file_path)
        return result["text"]

    # Réutiliser la fonction d'évaluation avec le modèle actuel
    results_by_model[model_name] = evaluate_combined_techniques(combined_results)

# Fonction pour la Randomisation Contrôlée (réduction de l'amplitude)
def controlled_randomization(audio, noise_level=0.001):
    # Ajouter un bruit gaussien minimal
    noise = np.random.normal(0, noise_level, len(audio))
    randomized_audio = audio + noise
    return randomized_audio

# Fonction pour la Pseudonymisation Avancée (réduction des modifications MFCC)
def advanced_pseudonymization(audio, sr, mfcc_noise_level=0.02):
    # Extraire les caractéristiques vocales (MFCC)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Modifier légèrement les MFCC pour créer une "pseudo-voix"
    modified_mfccs = mfccs + np.random.normal(0, mfcc_noise_level, mfccs.shape)

    # Reconstruire le signal à partir des MFCC modifiés
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(modified_mfccs, sr=sr)
    return reconstructed_audio

# Fusion des techniques avec ajustements
def combine_techniques(metadata):
    combined_results = []
    for idx, row in metadata.iterrows():
        audio = row["Preprocessed Audio"]
        sr = row["Normalized Sampling Rate"]

        # Étape 1 : Randomisation Contrôlée (réduction de l'amplitude)
        randomized_audio = controlled_randomization(audio, noise_level=0.001)

        # Étape 2 : Pseudonymisation Avancée (réduction des modifications MFCC)
        pseudonymized_audio = advanced_pseudonymization(randomized_audio, sr, mfcc_noise_level=0.02)

        # Stocker le résultat combiné
        combined_results.append({
            "File Path": row["File Path"],
            "Combined Audio": pseudonymized_audio
        })

    return pd.DataFrame(combined_results)

import os
import librosa
import numpy as np

# Importation des fonctions déjà définies dans votre pipeline
from your_module import (
    normalize_amplitude,
    remove_noise,
    apply_low_pass_filter,
    controlled_randomization,
    advanced_pseudonymization
)

def anonymize(input_audio_path):
    """
    Anonymization algorithm combining multiple techniques.
    
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
    
    # Étape 2 : Prétraitement
    audio = normalize_amplitude(audio)  # Normalisation de l'amplitude
    audio = remove_noise(audio, sr)  # Suppression du bruit
    audio = apply_low_pass_filter(audio, sr)  # Filtre passe-bas
    
    # Étape 3 : Anonymisation (Combinaison des techniques)
    audio = controlled_randomization(audio, noise_level=0.005)  # Randomisation Contrôlée
    audio = advanced_pseudonymization(audio, sr, mfcc_noise_level=0.05)  # Pseudonymisation Avancée
    
    # Étape 4 : Post-traitement (assurer la compatibilité avec soundfile.write)
    audio = audio.astype(np.float32)
    
    return audio, sr