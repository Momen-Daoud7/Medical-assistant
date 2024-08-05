import os
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
import re
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import pdfplumber
import re
import torch
import faiss
import numpy as np
import pickle
import logging
from tqdm import tqdm

# Ensure necessary NLTK data is downloaded
import nltk
nltk.download('punkt', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Initializing PDFProcessor with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.chunk_to_text = {}

    def extract_text_from_pdf(self, pdf_path):
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        logger.debug(f"Extracted {len(text)} characters from {pdf_path}")
        return text

    def clean_text(self, text):
        logger.debug("Cleaning extracted text")
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text

    def split_into_chunks(self, text, chunk_size=1000, overlap=100):
        logger.debug(f"Splitting text into chunks (size: {chunk_size}, overlap: {overlap})")
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def process_pdf(self, pdf_path):
        logger.info(f"Processing PDF: {pdf_path}")
        raw_text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = self.clean_text(raw_text)
        chunks = self.split_into_chunks(cleaned_text)
        logger.info(f"Processed {pdf_path}: {len(chunks)} chunks created")
        return chunks

    def get_embeddings(self, text_chunks):
        logger.info(f"Generating embeddings for {len(text_chunks)} chunks")
        embeddings = []
        for chunk in tqdm(text_chunks, desc="Embedding chunks", unit="chunk"):
            embedding = self.model.encode(chunk, show_progress_bar=False, convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
        return np.array(embeddings)

    def query(self, query_text, k=5):
        logger.info(f"Querying index with: '{query_text}'")
        query_embedding = self.model.encode([query_text], convert_to_tensor=True).cpu().numpy()
        
        if self.faiss_index is None:
            logger.error("FAISS index not loaded. Call load_index() first.")
            return []

        distances, indices = self.faiss_index.search(query_embedding, k)
        results = [self.chunk_to_text[i] for i in indices[0] if i in self.chunk_to_text]
        logger.info(f"Query returned {len(results)} results")
        return results


    def create_faiss_index(self, embeddings):
        logger.info("Creating Faiss index")
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        logger.info(f"Faiss index created with {index.ntotal} vectors of dimension {dimension}")
        return index

    def process_and_index_pdfs(self, pdf_directory):
        logger.info(f"Processing and indexing PDFs in directory: {pdf_directory}")
        all_chunks = []
        all_embeddings = []

        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")

        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            pdf_path = os.path.join(pdf_directory, filename)
            chunks = self.process_pdf(pdf_path)
            embeddings = self.get_embeddings(chunks)
            
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

        logger.info(f"Total chunks processed: {len(all_chunks)}")
        self.faiss_index = self.create_faiss_index(all_embeddings)
        self.chunk_to_text = {i: chunk for i, chunk in enumerate(all_chunks)}

    def save_index(self, index_path, chunk_to_text_path):
        logger.info(f"Saving Faiss index to {index_path}")
        faiss.write_index(self.faiss_index, index_path)
        logger.info(f"Saving chunk-to-text mapping to {chunk_to_text_path}")
        with open(chunk_to_text_path, 'wb') as f:
            pickle.dump(self.chunk_to_text, f)
        logger.info("Save completed")

    def load_index(self, index_path, chunk_to_text_path):
        logger.info(f"Loading Faiss index from {index_path}")
        self.faiss_index = faiss.read_index(index_path)
        logger.info(f"Loading chunk-to-text mapping from {chunk_to_text_path}")
        with open(chunk_to_text_path, 'rb') as f:
            self.chunk_to_text = pickle.load(f)
        logger.info("Load completed")

def extract_and_clean_text(pdf_path):
    cleaned_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Remove headers and footers (assume they're in the top/bottom 10% of the page)
            bbox = page.bbox
            crop_height = bbox[3] * 0.1
            cropped_page = page.crop((bbox[0], crop_height, bbox[2], bbox[3] - crop_height))
            text = cropped_page.extract_text()
            
            # Clean the text
            text = re.sub(r'-\n', '', text)  # Handle hyphenation
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special characters
            
            cleaned_text += text + "\n"
    
    return cleaned_text.strip()

def main():
    try:
        processor = PDFProcessor()

        # Process and index PDFs
        pdf_directory = "./Files"
        if not os.path.exists(pdf_directory):
            logger.error(f"PDF directory not found: {pdf_directory}")
            return

        processor.process_and_index_pdfs(pdf_directory)

        # Save the index and chunk-to-text mapping
        processor.save_index("pdf_knowledge_base.index", "chunk_to_text.pkl")

        # Load the index (simulating a separate run)
        processor.load_index("pdf_knowledge_base.index", "chunk_to_text.pkl")

        # Query example
        query = "How can the integrity and authenticity of DICOM objects be verified to prevent unauthorized modifications or tampering?"
        results = processor.query(query)
        logger.info(f"Top 5 relevant chunks for the query '{query}':")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result[:100]}...")  # Log first 100 characters of each chunk

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()