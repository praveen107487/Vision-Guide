from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import uuid
import base64
from gtts import gTTS
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)

# Create the 'static' folder if it doesn't exist
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Gemini API Configuration ---
# Your API key is placed directly in the code.
GEMINI_API_KEY = "AIzaSyCXq-ixIBjmrIJMmZL4za24EbAzuMU4l2A" 

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)


# --- AI Model Functions ---

def describe_image_with_gemini(image_path):
    """Generates a description for an image using the Gemini 2.0 Flash model."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return "Error: Gemini API key is not configured."
    try:
        # MODEL NAME UPDATED to current stable version
        model = genai.GenerativeModel('gemini-2.0-flash')
        image = Image.open(image_path)
        prompt = "Describe this image in a clear and concise way for a visually impaired person."
        response = model.generate_content([prompt, image])
        # Handle different response structures for Gemini API
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            # Try to get text from parts
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    return part.text
            return str(response.parts[0]) if response.parts else "Could not generate a description."
        else:
            return "Could not generate a description."
    except Exception as e:
        print(f"Error with Gemini Vision API: {e}")
        return f"Sorry, I couldn't describe the image. API Error: {e}"

def get_gemini_response(user_input):
    """Generates a chat response using the Gemini 2.0 Flash model."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return "Error: Gemini API key is not configured."
    try:
        # MODEL NAME UPDATED to current stable version.
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(user_input)

        # Handle different response structures for Gemini API
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            # Try to get text from parts
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    return part.text
            return str(response.parts[0]) if response.parts else "I am unable to respond right now."
        else:
            return "I am unable to respond right now."
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return f"Sorry, I encountered an error. API Error: {e}"


# --- Flask Routes (No changes below this line) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return "No image uploaded", 400
        file = request.files['image']
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        description = describe_image_with_gemini(filepath)
        if "Error:" in description or "Sorry," in description or "unable" in description:
             print(f"Description error: {description}")
             return "Could not get a description for the image.", 500

        tts = gTTS(description, lang='en')
        audio_filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        tts.save(audio_path)

        return redirect(url_for('result', image=filename, description=description, audio=audio_filename))
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image data found in request."}), 400
    try:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        description = describe_image_with_gemini(filepath)
        if "Error:" in description or "Sorry," in description or "unable" in description:
            os.remove(filepath)
            print(f"Description error: {description}")
            return jsonify({"error": "Could not get a description for the image."}), 500

        tts = gTTS(description, lang='en')
        audio_filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        tts.save(audio_path)
        
        result_url = url_for('result', image=filename, description=description, audio=audio_filename)
        return jsonify({'redirect': result_url})
    except Exception as e:
        print(f"An unexpected error occurred in /upload: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/result')
def result():
    image = request.args.get('image')
    description = request.args.get('description')
    audio = request.args.get('audio')
    return render_template('result.html', image=image, description=description, audio=audio)

@app.route('/gpt')
def gpt():
    return render_template('gpt.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get("message", "")
    reply = get_gemini_response(user_input)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)