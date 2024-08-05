import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from pdf_processor import PDFProcessor
import logging
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect
import openai


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Now you can access the environment variables using os.getenv()
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
AI71_API_KEY = os.getenv('AI71_API_KEY')
AI71_BASE_URL = os.getenv('AI71_BASE_URL')

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Configure AI71 client
# Initialize AI71 client


openai_client = openai.OpenAI(
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
)
# Initialize PDFProcessor
pdf_processor = PDFProcessor()
pdf_processor.load_index("pdf_knowledge_base.index", "chunk_to_text.pkl")

def estimate_tokens(text):
    # This is a rough estimate. For more accuracy, use a proper tokenizer.
    return len(text.split()) * 1.3  # Assumes average of 1.3 tokens per word

def retrieve_context(query, k=3, max_tokens=1500):
    relevant_chunks = pdf_processor.query(query, k)
    context = ""
    total_tokens = 0
    for chunk in relevant_chunks:
        chunk_tokens = estimate_tokens(chunk)
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += chunk + "\n\n"
        total_tokens += chunk_tokens
    logger.info(f"Retrieved context with approximately {total_tokens} tokens")
    return context


def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

def translate_text(text, source='auto', target='en'):
    try:
        translator = GoogleTranslator(source=source, target=target)
        return translator.translate(text)
    except:
        return text  # Return original text if translation fails
    

def generate_falcon_response(prompt, original_lang):
    try:
        # Translate prompt to English if it's not already
        if original_lang == 'ar':
            english_prompt = translate_text(prompt, source=original_lang, target='en')
        else:
            english_prompt = prompt

                       

        context = retrieve_context(english_prompt, max_tokens=1000)
        
        system_message = """You are an experienced X-ray technician helping colleagues troubleshoot machine issues. Respond naturally and conversationally, as if speaking to a coworker. Don't use phrases like 'based on my information' or 'my data'. Offer practical advice and ask follow-up questions if needed. Never end your response with 'User:' or any prompt for the user to speak."""

        few_shot_examples = """
        Q: How can I troubleshoot inconsistent or no output from the high-voltage generator?

A: Start by checking for loose connections or damaged cables. Then, verify the voltage setting and adjust if necessary. Make sure the power supply is stable and check the fuses – replace them if needed. Take a look at the system display for any fault codes or error messages. Also, double-check that the system is in the proper operating mode with correct settings. If you notice any physical damage to the generator, that could be the culprit. If you're still stuck after all this, it might be time to give the manufacturer a call for more specialized help.

Q: I'm experiencing intermittent image artifacts on our primary X-ray machine. They appear as dark spots or streaks on the image, and they're inconsistent. We've already checked for patient movement, electrical interference, and basic equipment issues. Any ideas on what might be causing these artifacts?

A: Hmm, intermittent artifacts can be tricky. Since you've already ruled out the basics, let's dig a bit deeper. Have you checked the imaging sensor or detector for any damage or debris? Sometimes even tiny particles can cause issues. It might be worth trying a different detector if you have one available – that could help isolate the problem. Don't forget to check the software side too. Any recent updates or glitches in the imaging software could be causing this. Also, consider environmental factors – any new equipment nearby that could be causing interference? If you're still stumped after trying these, it might be time to get the manufacturer involved. They often have more specialized diagnostic tools.

Q: Great, thanks for the additional information. We'll definitely check all connections again and consider testing the system in a different location. We'll also take a closer look at the detector for any signs of damage. Given that these artifacts are intermittent, could there be a possibility of a component that's overheating or experiencing thermal fluctuations causing the issue?

A: You're on the right track with that thinking! Overheating or thermal fluctuations could definitely cause intermittent artifacts. It's a good idea to monitor the temperature of the X-ray tube during operation. Is it staying within the manufacturer's recommended range? If it's getting too hot, it could affect image quality or even damage the tube. While you're at it, take a look at the cooling system. Make sure fans are running properly and air vents aren't blocked. Sometimes, a failing cooling component can cause these kinds of intermittent issues. If you have a thermal camera or even an infrared thermometer, it might be worth scanning the machine during operation to look for any unexpected hot spots. Keep me posted on what you find – we'll get to the bottom of this!
"""
        

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": few_shot_examples},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {english_prompt}"}
        ]
        
        total_tokens = sum(estimate_tokens(m['content']) for m in messages)
        logger.info(f"Estimated total input tokens: {total_tokens}")
        
        response = openai_client.chat.completions.create(
            model="tiiuae/falcon-180B-chat",
            messages=messages,
            max_tokens=min(2048 - total_tokens, 500),
            temperature=0.7,
        )
        
        logger.info("Generated Falcon-enhanced response")
        response_text = response.choices[0].message.content.strip()
        
        # Remove "User:" if it appears at the end of the response
        response_text = response_text.rstrip()
        if response_text.endswith("User:"):
            response_text = response_text[:-5].rstrip()
        
        # Translate response back to original language if necessary
        if original_lang == 'ar':
            response_text = translate_text(response_text, source='en', target="ar")
        
        return response_text
    except Exception as e:
        logger.error(f"Error generating Falcon response: {e}", exc_info=True)
        return "I'm sorry, I couldn't process that request. Please try again later."

@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From')

    logger.info(f"Received message from {sender}: {incoming_msg}")

    # Detect the language of the incoming message
    detected_lang = detect_language(incoming_msg)
    logger.info(f"Detected language: {detected_lang}")

    response = generate_falcon_response(incoming_msg, detected_lang)
    logger.info(f"Generated response: {response[:100]}...")  # Log first 100 chars

    twiml = MessagingResponse()
    twiml.message(response)
    return str(twiml)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)



