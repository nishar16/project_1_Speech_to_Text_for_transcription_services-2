# project_1_Speech_to_Text_for_transcription_services-2


---

### ✅ **Environment Setup**

```python
!pip install kaggle
!pip install transformers torchaudio librosa noisereduce
!pip install openai-whisper  # Optional for using Whisper
!pip install datasets
```

---

### 📁 **Data Upload**

```python
from google.colab import files
files.upload()  # User uploads the `.tar.gz` audio dataset
```

---

### 🎧 **Module 1: Data Cleaning**

* **Tasks:**

  * Load the audio file (`librosa.load`)
  * Trim silence (`librosa.effects.trim`)
  * Normalize the waveform (`librosa.util.normalize`)
  * Visualize cleaned waveform using `matplotlib`

```python
audio_path = '/sp01_street_sn5.wav'
y, sr = librosa.load(audio_path, sr=None)
y_clean, _ = librosa.effects.trim(y)
y_normalized = librosa.util.normalize(y_clean)

# Visualization
librosa.display.waveshow(y_normalized, sr=sr)
```

---

### 📊 **Module 2: Data Analysis**

* Plotting the **spectrogram** to visualize noise, accents, and clarity of the audio.

```python
D = librosa.amplitude_to_db(np.abs(librosa.stft(y_normalized)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
```

---

### 📈 **Module 3: Model Evaluation**

* **Metric Used:** Word Error Rate (WER) with `jiwer`

```python
!pip install jiwer
from jiwer import wer

# Example usage
actual_transcription = "your actual transcription"
predicted_transcription = "your model output"
error = wer(actual_transcription, predicted_transcription)
```

---
