import os
import warnings
import librosa
import numpy as np
import asyncio
import sqlite3
import io
import datetime
from pydub import AudioSegment
import soundfile as sf
from scipy.io import wavfile
from flask import Flask, request, render_template_string
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
from werkzeug.middleware.proxy_fix import ProxyFix

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define classes and initialize label encoder
classes = ['Snoring', 'No-snoring', 'Male-snoring', 'Female-snoring']
labelencoder = LabelEncoder()
labelencoder.fit(classes)

# Load model
model_path = 'saved_models/audio_classification18_90(1).keras'
model = load_model(model_path)


DB_PATH = 'snore_audio.db'

def save_prediction_to_db(file_name, classification, intensity, frequency, snore_index, consistency):
    try:
        # Connect to SQLite3 database
        connection = sqlite3.connect(DB_PATH)
        cursor = connection.cursor()
        
        # Ensure the table exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS snoring_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            classification TEXT,
            intensity REAL,
            frequency REAL,
            snore_index TEXT,
            consistency TEXT
        );
        """)
        
        sql = """
        INSERT INTO snoring_predictions 
        (file_name, classification, intensity, frequency, snore_index, consistency)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        cursor.execute(sql, (file_name, classification, intensity, frequency, snore_index, consistency))
        # cursor.execute("SELECT * FROM snoring_predictions")
        # result = cursor.fetchall()
        # print (result)
        connection.commit()

    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        connection.close()


# Feature extraction with threading
def extract_features(frames, sample_rate, n_mfcc=30):
    def process_frame(frame):
        mfccs_features = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfccs_features)
        delta2_mfcc = librosa.feature.delta(mfccs_features, order=2)
        combined_features = np.concatenate((mfccs_features, delta_mfcc, delta2_mfcc), axis=0)
        return np.mean(combined_features.T, axis=0).reshape(1, -1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        features = list(executor.map(process_frame, frames))
    return np.vstack(features)

# Asynchronous batch prediction
async def async_predict(frames, sample_rate, model, batch_size=32):
    predictions = []

    async def process_batch(batch):
        loop = asyncio.get_event_loop()
        batch_features = await loop.run_in_executor(None, extract_features, batch, sample_rate)
        batch_predictions = await loop.run_in_executor(None, model.predict, batch_features)
        return np.argmax(batch_predictions, axis=1)

    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(*(process_batch(batch) for batch in batches))
    for batch_result in results:
        predictions.extend(batch_result)

    return predictions

def analyze_snore_consistency(audio, sample_rate, model, frame_duration=0.4, frame_overlap=0.2, max_gap_threshold=4.0):
    frame_size = int(frame_duration * sample_rate)
    hop_size = int(frame_size * (1 - frame_overlap))
    frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size).T

    predictions = asyncio.run(async_predict(frames, sample_rate, model))
    snoring_frames = []
    is_snoring = False

    for i, pred_idx in enumerate(predictions):
        prediction_class = labelencoder.inverse_transform([pred_idx])[0]
        current_time = i * (frame_duration * (1 - frame_overlap))
        if prediction_class in ['Snoring', 'Male-snoring', 'Female-snoring']:
            if not is_snoring:
                is_snoring = True
                snoring_frames.append(current_time)
        else:
            is_snoring = False

    if snoring_frames:
        gaps = np.diff(snoring_frames)
        max_gap = max(gaps, default=0)
        return "Regular" if max_gap <= max_gap_threshold else "Irregular"
    else:
        return "Irregular: No snoring detected"

def calculate_snore_index(intensity, frequency):
    normalized_intensity = np.clip((intensity - 50) / 50, 0, 1) * 100
    normalized_frequency = np.clip((frequency - 20) / 300, 0, 1) * 100
    snore_index = (normalized_intensity * 0.4 + normalized_frequency * 0.3) 
    return np.clip(snore_index, 0, 100)

def classify_snore_index(snore_index):
    if snore_index < 10.33:
        return "Mild"
    elif snore_index < 30:
        return "Moderate"
    else:
        return "Extreme"

def analyze_audio_directly(audio_binary):
    print(type(audio_binary), "analyse_audio_directly")
    try:
        audio_buffer = io.BytesIO(audio_binary)
        audio, sample_rate = librosa.load(audio_buffer, res_type='kaiser_fast')
            
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        duration = len(audio) / sample_rate
        if duration < 10.0:
            return "Error: Audio length must be at least 10 seconds."
        if duration > 30.0:
            return "Error: Audio length must not exceed 30 seconds."

        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=30)
        delta_mfcc = librosa.feature.delta(mfccs_features)
        delta2_mfcc = librosa.feature.delta(mfccs_features, order=2)
        combined_features = np.concatenate((mfccs_features, delta_mfcc, delta2_mfcc), axis=0)
        mfccs_scaled_features = np.mean(combined_features.T, axis=0).reshape(1, -1)

        predicted_probabilities = model.predict(mfccs_scaled_features, verbose=0)
        predicted_label = np.argmax(predicted_probabilities, axis=1)
        prediction_class = labelencoder.inverse_transform(predicted_label)[0]

        if prediction_class == "Snoring":
            prediction_class = "Male-snoring"

        result = {
            'classification': prediction_class,
            'audio_data': audio_binary
        }

        if prediction_class != 'No-snoring':
            rmse = librosa.feature.rms(y=audio)
            rmse_db = librosa.amplitude_to_db(rmse, ref=np.max)
            average_intensity = np.mean(rmse_db)
            target_dB = 60
            intensity = average_intensity + target_dB

            stft = np.abs(librosa.stft(audio))
            freqs = librosa.fft_frequencies(sr=sample_rate)
            power_spectrum = np.sum(stft ** 2, axis=1)
            frequency = freqs[np.argmax(power_spectrum)]

            snore_index = calculate_snore_index(intensity, frequency)
            severity = classify_snore_index(snore_index)
            consistency = analyze_snore_consistency(audio, sample_rate, model)

            result.update({
                'intensity': round(intensity, 2),
                'frequency': round(frequency, 2),
                'snore_index': severity,
                'consistency': consistency
            })

        return result

    except Exception as e:
        return f"Error processing audio: {str(e)}"

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config['WTF_CSRF_ENABLED'] = True
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024  

# Create directories for uploads and saved files
UPLOAD_FOLDER = 'uploads'
SAVED_FOLDER = 'saved_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVED_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    html_template = """
   <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Snoring Detection dev</title>
        <style>
            /* Keep all existing styles unchanged */
            body {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #e9ecef;
                color: #333;
            }

            h1 {
                margin-bottom: 20px;
                font-size: 2.5em;
            }

            h2 {
                font-size: 1.5em;
                margin-bottom: 15px;
                color: #555;
            }

            .container {
                display: flex;
                flex-wrap: wrap;
                gap: 30px;
                justify-content: center;
                align-items: flex-start;
                padding: 20px;
            }

            .section {
                background-color: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                width: 300px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            form {
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 100%;
            }

            input[type="file"] {
                margin-bottom: 15px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 8px;
                width: 100%;
            }

            button {
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                margin-top: 10px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1em;
                transition: background-color 0.3s ease;
            }

            button:hover {
                background-color: #0056b3;
            }

            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }

            #loader {
                display: none;
                margin: 20px auto;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007BFF;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }

            audio {
                margin-top: 10px;
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 8px;
            }

            pre {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #ddd;
                text-align: left;
                font-size: xx-large;
                white-space: pre-wrap;
                animation: fadeIn 1s;
                opacity: 1;
            }

            pre.show {
                opacity: 1;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .popup {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                padding: 20px;
                width: 80%;
                max-width: 500px;
                z-index: 1000;
            }

            .popup-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                font-weight: bold;
            }

            .close-btn {
                background: none;
                border: none;
                font-size: 1.5em;
                font-weight: bold;
                cursor: pointer;
                color: #333;
            }

            .close-btn:hover {
                color: red;
            }

            .show {
                display: block;
            }

            .hide {
                display: none;
            }

            #overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 999;
                display: none;
            }

            #overlay.show {
                display: block;
            }
        </style>
        
    </head>
    <body>
        <h1>Snoring Detection</h1>
        <h3>Note: <span style="font-size: 0.9em; font-weight: normal;">Audio should be between 10 to 30 seconds long for accurate analysis.</span></h3>
        <div class="container">
            <div class="section">
                <h2>Record Audio</h2>
                <button id="record-btn">Start Recording</button>
                <button id="stop-btn" disabled>Stop Recording</button>
                <p>Recording Duration: <span id="recording-timer">00:00</span></p>
                <audio id="audio-playback" style="display:none" controls></audio>
                <form id="recording-form" method="POST" enctype="multipart/form-data">
                    <button type="submit" id="analyze-btn" disabled>Analyze Recording</button>
                </form>
            </div>
        </div>
        <div id="loader"></div>
        {% if result %}
        <div id="popup" class="popup show">
            <div class="popup-header">
                <span>Result</span>
                <button class="close-btn" onclick="closePopup()">×</button>
            </div>
            <pre id="result">{{ result }}</pre>
        </div>
        {% endif %}
        <pre id="result" class="hide"></pre>

            <script>
                let mediaRecorder;
                let audioChunks = [];
                let recordingTimer;
                let secondsElapsed = 0;

                let audioContext;
                let audioStream;
                let gainNode;
                let mediaStreamDestination;
                let currentStream = null;

                const recordButton = document.getElementById("record-btn");
                const stopButton = document.getElementById("stop-btn");
                const audioPlayback = document.getElementById("audio-playback");
                const analyzeButton = document.getElementById("analyze-btn");
                const recordingForm = document.getElementById("recording-form");
                const recordingTimerDisplay = document.getElementById("recording-timer");

                async function submitForm(formData) {
                console.log("Submitting form data")
                    document.getElementById('loader').style.display = 'block';
                    try {
                        const response = await fetch(window.location.href, {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const result = await response.text();
                        
                        // Parse the response to extract just the result
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(result, 'text/html');
                        const resultText = doc.querySelector('#result')?.textContent;
                        
                        if (resultText) {
                            // Create and show popup with result
                            console.log("Server response received:", resultText);
                            const popupHTML = `
                                <div id="popup" class="popup show">
                                    <div class="popup-header">
                                        <span>Result</span>
                                        <button class="close-btn" onclick="closePopup()">×</button>
                                    </div>
                                    <pre id="result">${resultText}</pre>
                                </div>`;
                            
                            // Remove existing popup if any
                            const existingPopup = document.getElementById('popup');
                            if (existingPopup) {
                                existingPopup.remove();
                            }
                            
                            // Add new popup
                            document.body.insertAdjacentHTML('beforeend', popupHTML);
                        }
                        
                    } catch (error) {
                        console.error('Error:', error);
                        alert('Error uploading recording: ' + error.message);
                    } finally {
                        document.getElementById('loader').style.display = 'none';
                    }
                }

                function resetRecorder() {
                    console.log("Resetting recorder...");
                    secondsElapsed = 0;
                    recordingTimerDisplay.textContent = "00:00";
                    recordButton.disabled = false;
                    stopButton.disabled = true;
                    analyzeButton.disabled = true;

                    if (mediaRecorder && mediaRecorder.state !== "inactive") {
                        console.log("Stopping MediaRecorder...");
                        mediaRecorder.stop();
                    }

                    // Stop all tracks in the current stream
                    if (currentStream) {
                        console.log("Stopping all audio tracks...");
                        currentStream.getTracks().forEach(track => track.stop());
                        currentStream = null;
                    }

                    // Clear audio chunks
                    audioChunks = [];

                    // Reset audio context and related objects
                    if (audioContext && audioContext.state !== 'closed') {
                        console.log("Closing audio context...");
                        audioStream?.disconnect();
                        gainNode?.disconnect();
                        mediaStreamDestination?.disconnect();
                        audioContext.close();
                    }

                    audioContext = null;
                    audioStream = null;
                    gainNode = null;
                    mediaStreamDestination = null;
                    mediaRecorder = null;
                }

                recordButton.addEventListener("click", async () => {
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        alert("Your browser does not support audio recording.");
                        return;
                    }

                    try {
                        console.log("Starting audio recording...");
                        resetRecorder();

                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            audio: {}
                        });

                        currentStream = stream; 
                        audioContext = new AudioContext();
                        audioStream = audioContext.createMediaStreamSource(stream);
                        gainNode = audioContext.createGain();
                        mediaStreamDestination = audioContext.createMediaStreamDestination();
                        gainNode.gain.value = 1.0;

                        audioStream.connect(gainNode);
                        gainNode.connect(mediaStreamDestination);
                        mediaRecorder = new MediaRecorder(stream);

                        audioChunks = [];
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                            console.log(event.data, "Type of Recoreded Audio file")
                        };

                        mediaRecorder.onstop = async () => {
                            clearInterval(recordingTimer);

                            if (secondsElapsed < 10) {
                                alert("Recording must be at least 10 seconds long. Keep going.");
                                resetRecorder();
                                return;
                            }

                            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                            const audioURL = URL.createObjectURL(audioBlob);
                            audioPlayback.src = audioURL;

                            const audioFile = new File([audioBlob], `recording_${Date.now()}.wav`, {
                                type: 'audio/wav'
                            });

                            const formData = new FormData();
                            formData.append("audiofile", audioFile);

                            recordingForm.onsubmit = async (e) => {
                                e.preventDefault();
                                await submitForm(formData);
                            };

                            analyzeButton.disabled = false;
                            recordButton.disabled = false;

                            console.log("Audio file created:");
                        };

                        mediaRecorder.start();
                        recordButton.disabled = true;
                        stopButton.disabled = false;

                        recordingTimer = setInterval(() => {
                            secondsElapsed++;
                            const minutes = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
                            const seconds = String(secondsElapsed % 60).padStart(2, "0");
                            recordingTimerDisplay.textContent = `${minutes}:${seconds}`;

                            if (secondsElapsed >= 30) {
                                mediaRecorder.stop();
                                alert("Recording must be at most 30 seconds long.");
                            }
                        }, 1000);
                    } catch (error) {
                        console.error("Error accessing microphone:", error);
                        alert("Error accessing your microphone. Please ensure it is enabled.");
                    }
                });

                stopButton.addEventListener("click", () => {
                    console.log("Stop button clicked. Stopping recording...");
                    if (mediaRecorder && mediaRecorder.state === "recording") {
                        mediaRecorder.stop();
                        clearInterval(recordingTimer);
                        stopButton.disabled = true;
                    } else {
                        console.warn("MediaRecorder is not in a recording state.");
                    }
                });

                function closePopup() {
                    const popup = document.getElementById('popup');
                    if (popup) {
                        popup.classList.add('hide');
                        popup.classList.remove('show');
                    }
                    resetRecorder();
                }
            </script>
    </body>
    </html>
    """

    def convert_to_wav(audio_binary):
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_binary))
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            return wav_io.read()
        except Exception as e:
            raise ValueError(f"Audio conversion to WAV failed: {e}")

    import datetime
    if request.method == 'POST':
        try:
            if 'audiofile' not in request.files:
                return render_template_string(html_template, result="Error: No file provided")
                
            file = request.files['audiofile']
            print(type(file))
            if not file:
                return render_template_string(html_template, result="Error: No file selected")

            audio_binary = file.read()
            filename = file.filename
            
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_binary))
                audio = audio.set_frame_rate(16000)
                
                # Apply gain (4.0 = +12dB)
                audio = audio + 12
                
                # Export as WAV
                wav_io = io.BytesIO()
                audio.export(wav_io, format='wav')
                audio_binary = wav_io.getvalue()
                filename = filename.rsplit('.', 1)[0] + ".wav"
            except ValueError as e:
                return render_template_string(html_template, result=f"Error: {str(e)}")

            # Analyze audio directly
            result = analyze_audio_directly(audio_binary)
            
            if isinstance(result, dict):
                # Save file only after successful analysis
                file_path = os.path.join(SAVED_FOLDER, filename)
                with open(file_path, 'wb') as f:
                    f.write(audio_binary)
                
                # Save to database
                save_prediction_to_db(
                    file_name=filename,
                    classification=result['classification'],
                    intensity=result.get('intensity'),
                    frequency=result.get('frequency'),
                    snore_index=result.get('snore_index', 'N/A'),
                    consistency=result.get('consistency', 'N/A')
                )
                

                display_result = "\n"     
                if result['classification'] == 'No-snoring':
                    display_result = f"Classification: {result['classification']}\n"
                
                
                # display_result = f"Classification: {result['classification']}\n"
                if 'intensity' in result:
                    display_result += f"Intensity (dB): {result['intensity']}\n"
                    display_result += f"Frequency (Hz): {result['frequency']}\n"
                    display_result += f"Snore Index: {result['snore_index']}\n"
                    display_result += f"Consistency: {result['consistency']}"
                
                return render_template_string(html_template, result=display_result)
            else:
                return render_template_string(html_template, result=result)

        except Exception as e:
            return render_template_string(html_template, result=f"Error: {str(e)}")

    return render_template_string(html_template, result=None)

if __name__ == '__main__':
    app.run(debug=True)
