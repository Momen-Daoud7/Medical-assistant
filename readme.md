# X-ray Machine Troubleshooting Chatbot

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Code Documentation](#code-documentation)
- [Development Challenges and Solutions](#development-challenges-and-solutions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Support](#support)

## Overview
The X-ray Machine Troubleshooting Chatbot is an AI-powered assistant designed to help X-ray technicians and engineers quickly diagnose and resolve issues with X-ray machines. By leveraging advanced language models and a comprehensive knowledge base, this chatbot provides instant, accurate troubleshooting advice through WhatsApp, supporting multiple languages for global accessibility.

## Problem Statement

X-ray machine technicians and engineers often face complex troubleshooting scenarios that require quick access to specialized knowledge. Traditional methods of consulting manuals or waiting for expert assistance can lead to prolonged downtime and reduced efficiency in medical facilities.

## Solution

We've developed an AI-powered chatbot that leverages the Falcon language model to provide instant, accurate troubleshooting assistance for X-ray machines. This chatbot combines:

1. Retrieval-Augmented Generation (RAG) for context-aware responses
2. Multi-language support for global accessibility
3. Integration with WhatsApp for convenient access

## Features

- **AI-Powered Responses**: Utilizes the Falcon 180B model for generating human-like, context-aware troubleshooting advice.
- **Knowledge Base Integration**: Incorporates information from technical documentation to provide accurate and up-to-date assistance.
- **Multi-Language Support**: Detects the language of incoming queries and provides responses in the same language.
- **WhatsApp Integration**: Accessible through WhatsApp for easy and quick communication.
- **Natural Conversation Flow**: Engages in a conversational manner, making it user-friendly for technicians.

## Technical Stack

- **AI Model**: Falcon 180B via AI71 API
- **RAG Implementation**: Custom PDF processor for context retrieval
- **Language Processing**: deep_translator for translation, langdetect for language detection
- **Backend**: Flask
- **Messaging Platform**: Twilio (WhatsApp API)

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/x-ray-troubleshooting-bot.git
   cd x-ray-troubleshooting-bot
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following:
   ```
   TWILIO_ACCOUNT_SID=your_twilio_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_whatsapp_number
   AI71_API_KEY=your_ai71_api_key
   ```

5. Prepare your knowledge base:
   - Place your X-ray machine technical PDFs in the `./Files` directory
   - Run the PDF processor script to create the knowledge base index:
     ```
     python pdf_processor.py
     ```

6. Start the Flask server:
   ```
   python app.py
   ```

7. Set up Twilio Webhook:
   - Go to your Twilio Console
   - Set the Webhook URL for incoming messages to `http://your-server-address/webhook`

## Usage

Once set up, technicians can send messages to the designated WhatsApp number. The chatbot will:

1. Receive the message
2. Detect the language
3. Translate to English if necessary
4. Retrieve relevant context from the knowledge base
5. Generate a response using the Falcon model
6. Translate the response back to the original language if needed
7. Send the response via WhatsApp

Example interaction:

User: "The X-ray tube is overheating after 10 minutes of use. What should I check?"

Bot: "Overheating of the X-ray tube could be serious. First, shut down the machine for safety. Check the cooling system - ensure the coolant levels are adequate and the cooling fans are functioning properly. Also, verify that the power supply is stable and within the correct range. If you notice any unusual odors or visible damage, don't restart the machine. Instead, contact the manufacturer's support team immediately. Once you've checked these, if everything seems normal, try a short test run while closely monitoring the temperature. If the problem persists, you may need to have the tube inspected by a specialist."

## Code Documentation

This section provides an overview of the main components and functions in the X-ray Machine Troubleshooting Chatbot.

### Main Application (`app.py`)

The main application file contains the Flask server and the core logic for processing messages and generating responses.

#### Key Functions:

- `webhook()`: Handles incoming WhatsApp messages via Twilio webhook.
  ```python
  @app.route('/webhook', methods=['POST'])
  def webhook():
      incoming_msg = request.values.get('Body', '').strip()
      sender = request.values.get('From')
      # Process message and generate response
      # ...
  ```

- `generate_falcon_response(prompt, original_lang)`: Generates a response using the Falcon model.
  ```python
  def generate_falcon_response(prompt, original_lang):
      # Translate prompt if necessary
      # Retrieve context
      # Generate response using Falcon model
      # Translate response back if necessary
      # ...
  ```

- `detect_language(text)`: Detects the language of the input text.
  ```python
  def detect_language(text):
      try:
          return detect(text)
      except:
          return 'en'  # Default to English if detection fails
  ```

- `translate_text(text, source='auto', target='en')`: Translates text between languages.
  ```python
  def translate_text(text, source='auto', target='en'):
      try:
          translator = GoogleTranslator(source=source, target=target)
          return translator.translate(text)
      except:
          return text  # Return original text if translation fails
  ```

### PDF Processor (`pdf_processor.py`)

This module handles the processing of PDF documents and creation of the knowledge base.

#### Key Classes and Functions:

- `class PDFProcessor`: Manages PDF processing and indexing.
  ```python
  class PDFProcessor:
      def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
          # Initialize embeddings model, FAISS index, etc.
          # ...

      def process_pdf(self, pdf_path):
          # Extract text from PDF
          # Clean and chunk text
          # ...

      def create_faiss_index(self, embeddings):
          # Create FAISS index from embeddings
          # ...

      def query(self, query_text, k=5):
          # Perform similarity search in FAISS index
          # ...
  ```

- `extract_text_from_pdf(pdf_path)`: Extracts text content from a PDF file.
  ```python
  def extract_text_from_pdf(pdf_path):
      # Use PyMuPDF to extract text from PDF
      # ...
  ```

- `clean_text(text)`: Cleans extracted text by removing special characters and normalizing whitespace.
  ```python
  def clean_text(text):
      # Remove special characters and normalize whitespace
      # ...
  ```

### Utility Functions (`utils.py`)

This file contains various utility functions used throughout the project.

#### Key Functions:

- `estimate_tokens(text)`: Estimates the number of tokens in a given text.
  ```python
  def estimate_tokens(text):
      # Rough estimation of tokens based on word count
      return len(text.split()) * 1.3
  ```

- `load_environment_variables()`: Loads environment variables from a .env file.
  ```python
  def load_environment_variables():
      load_dotenv()
      # Check if all required variables are set
      # ...
  ```

### Configuration (`config.py`)

This file contains configuration settings for the application.

```python
# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# AI71 Configuration
AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = os.getenv('AI71_API_KEY')

# Application Settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
MAX_TOKENS = 2048
TEMPERATURE = 0.7
```

### Testing (`tests/`)

The `tests/` directory contains unit tests for various components of the application.

- `test_pdf_processor.py`: Tests for the PDF processing functionality.
- `test_translation.py`: Tests for language detection and translation functions.
- `test_falcon_integration.py`: Tests for integration with the Falcon model.

To run the tests:
```
python -m pytest tests/
```

For more detailed information on each module and function, please refer to the inline documentation in the respective Python files.

## Development Challenges and Solutions

During the development of the X-ray Machine Troubleshooting Chatbot, we encountered several significant challenges. Here's how we addressed each one:

### 1. Limited Context Window

**Challenge:** The Falcon 180B model has a context window of 2048 tokens, which limits the amount of information we can provide for complex troubleshooting scenarios.

**Solution:** We implemented a context optimization strategy:
- Developed a token estimation function to gauge the length of inputs.
- Created a dynamic context retrieval system that adjusts the amount of context based on the query length.
- Implemented a priority system for context chunks, ensuring the most relevant information is included within the token limit.
- Fine-tuned prompts to be concise yet informative, maximizing the use of available tokens.

This approach allows us to provide rich, relevant context for each query while staying within the model's token limit.

### 2. Multi-Language Support

**Challenge:** Supporting multiple languages could potentially increase API calls to the Falcon model, leading to higher costs and longer response times.

**Solution:** We implemented a two-step translation process:
- Utilized the `deep_translator` library for translating user queries to English before processing.
- Used the same library to translate Falcon's English responses back to the user's original language.
- Implemented language detection using the `langdetect` library to automatically identify the input language.

This solution allows us to:
- Maintain a single English knowledge base.
- Minimize API calls to the Falcon model.
- Provide multilingual support without increasing the load on our primary AI model.

### 3. RAG with Multiple PDFs

**Challenge:** Implementing Retrieval-Augmented Generation (RAG) with a large corpus of PDF documents posed challenges in terms of efficient storage, quick retrieval, and relevant context selection.

**Solution:** We developed a comprehensive PDF processing and indexing system:
- Created a PDF processor that extracts, cleans, and chunks text from multiple PDFs.
- Implemented a vector database using FAISS for efficient storage and similarity search of text chunks.
- Developed a custom ranking algorithm to prioritize the most relevant chunks for each query.
- Implemented periodic reindexing to keep the knowledge base up-to-date with the latest documentation.

This system allows us to:
- Efficiently process and store information from a large number of technical PDFs.
- Quickly retrieve relevant context for user queries.
- Easily update the knowledge base as new documentation becomes available.

### 4. Handling Technical Jargon

**Challenge:** X-ray machine troubleshooting involves highly technical terms that general-purpose language models might not handle accurately.

**Solution:** We enhanced the model's understanding of technical jargon by:
- Fine-tuning the Falcon model on a dataset of X-ray machine technical documentation.
- Implementing a custom dictionary of X-ray-specific terms and their explanations.
- Developing prompt templates that encourage the model to use and explain technical terms accurately.

This approach significantly improved the accuracy and relevance of the chatbot's responses when dealing with specialized X-ray machine terminology.

### 5. Ensuring Real-time Performance

**Challenge:** Combining multiple processes (language detection, translation, RAG, and AI inference) while maintaining quick response times was challenging.

**Solution:** We optimized the system's performance through:
- Implementing asynchronous processing where possible, especially for I/O-bound operations like translation and database queries.
- Utilizing caching mechanisms for frequently asked questions and their translations.
- Optimizing the RAG process to minimize the number of database queries.
- Implementing a fallback mechanism to provide general responses if the full process exceeds a certain time threshold.

These optimizations allow the chatbot to provide helpful responses in near real-time, enhancing the user experience for X-ray technicians needing quick assistance.

## Contributing

We welcome contributions to improve the chatbot's knowledge base, enhance its troubleshooting capabilities, or extend its features. Please see our CONTRIBUTING.md file for guidelines on how to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- AI71 for providing access to the Falcon 180B model
- Twilio for WhatsApp integration capabilities
- All contributors and maintainers of the open-source libraries used in this project

## Support

For any questions or issues, please open an issue in the GitHub repository or contact our support team at momendaoud.soft@gmail.com.