import re
from io import BytesIO
from docx import Document
from fpdf import FPDF
import fitz  # PyMuPDF
import os
import openai
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
# Load API key from environment variable

# --- App Design Enhancements ---

# --- Function to display the pricing page ---
def show_pricing_page():
    st.title("Options d'abonnement")

    st.markdown(
        """
        ## Choisissez l'abonnement qui vous convient :

        ### Abonnement hebdomadaire : 15$

        * **6 résumés autorisés par semaine.**
        * Idéal pour les étudiants qui ont besoin de résumer quelques cours chaque semaine.
        * Accès à toutes les fonctionnalités de base.

        ### Abonnement mensuel : 50$

        * **20 résumés autorisés par mois.**
        * Parfait pour les étudiants ou professionnels qui ont besoin de résumer un grand nombre de documents.
        * Accès à toutes les fonctionnalités, y compris les fonctionnalités premium (à venir).

        ---

        **Avantages des abonnements :**

        * **Gain de temps considérable :** Obtenez des résumés concis et précis en quelques minutes.
        * **Meilleure compréhension :** Concentrez-vous sur les points clés de vos documents.
        * **Amélioration de la productivité :** Traitez plus d'informations en moins de temps.

        **Prêt à vous abonner ?**

        Cliquez sur le bouton ci-dessous pour choisir votre abonnement et commencer à profiter de tous les avantages !

        """
    )

# --- Sidebar ---
st.sidebar.title("Options de paiement")
st.sidebar.write("Prix par résumé : 5$")  # Display price per summary

if st.sidebar.button("Obtenir un abonnement"):
    show_pricing_page()  # Show the pricing page

# --- Main Content ---
st.title("Choix du LLM")

# --- Get OpenAI API key from user input ---
openai_api_key = st.text_input("Entrez votre clé API OpenAI :", type="password")

if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Veuillez entrer votre clé API OpenAI pour continuer.")
    st.stop()  # Stop execution if no API key is provided


if openai_api_key is None:
    st.error("OpenAI API key not found. Please set it as an environment variable.")

openai.api_key = openai_api_key
# Constants# Constants
MAX_TOKENS_PER_REQUEST = 4096  # A safer limit to avoid hitting the token cap
PDF_CHUNK_SIZE = 1000  # Smaller chunk size for faster PDF processing
DOCX_CHUNK_SIZE = 3000  # Larger chunk size for DOCX files
MAX_WORKERS = 4  # Number of threads for parallel processing
MAX_CHUNK_TOKENS = 2048  # Maximum token limit per chunk (for safety)

# Function to read a .docx file and return a list of paragraphs
def read_docx(file):
    doc = Document(file)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip() != ""]
    return paragraphs

# Optimized function to read a PDF file and return a list of paragraphs using fitz (PyMuPDF)
def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    
    # Iterate through the PDF pages and extract text
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    
    # Split the text into paragraphs
    paragraphs = [para for para in text.split('\n') if para.strip() != ""]
    return paragraphs

# Preprocess text for PDFs to remove unwanted elements and retain paragraph structure
def preprocess_paragraphs(paragraphs):
    cleaned_paragraphs = []
    for para in paragraphs:
        para = re.sub(r"\b\w+\s\d{4}\b", "", para)  # Remove "Name Year" patterns
        para = re.sub(r"\b\d+\b", "", para)         # Remove standalone numbers
        para = re.sub(r"\s+", " ", para).strip()    # Remove excess whitespace
        if len(para.split()) > 4:  # Only keep meaningful paragraphs
            cleaned_paragraphs.append(para)
    return cleaned_paragraphs

# Function to split the text into chunks for summarization
def split_into_chunks(paragraphs, max_chars):
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += " " + para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to estimate the number of tokens in a given text (approximate calculation)
def estimate_token_count(text):
    return len(text.split()) * 1.3  # Each word is roughly 1.3 tokens on average

# Function to recursively split a chunk if it's too large
def split_large_chunk(chunk, max_chunk_tokens=MAX_CHUNK_TOKENS):
    if estimate_token_count(chunk) > max_chunk_tokens:
        # Split the chunk in half and process recursively
        mid_point = len(chunk) // 2
        split_point = chunk[:mid_point].rfind(" ")  # Split at nearest space
        part1, part2 = chunk[:split_point], chunk[split_point:]
        return split_large_chunk(part1, max_chunk_tokens) + split_large_chunk(part2, max_chunk_tokens)
    return [chunk]

# Function to summarize a chunk of text, ensuring it doesn't exceed token limits
def summarize_chunk(chunk):
    # Recursively split the chunk if it's too large
    sub_chunks = split_large_chunk(chunk)
    summaries = []
    
    for sub_chunk in sub_chunks:
        prompt = (
            f"Vous êtes un assistant pédagogique chargé de condenser un cours de droit. "
            f"Faites un résumé concis et fluide en regroupant les idées dans des paragraphes cohérents. "
            f"Conservez uniquement les points clés essentiels et faites un résumé exhaustif du texte suivant :\n\n"
            f"Texte à résumer :\n\n{sub_chunk}"
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1800,
                temperature=0.3,
            )
            summaries.append(response.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")
            return None

    return "\n\n".join(summaries)

# Function to clean and format the output text
def clean_output(text):
    cleaned_text = re.sub(r'\bRésumé[: ]*', '', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to summarize the entire document (for PDFs) with parallel processing
def summarize_entire_document_pdf(paragraphs):
    paragraphs = preprocess_paragraphs(paragraphs)  # Clean the paragraphs
    chunks = split_into_chunks(paragraphs, max_chars=PDF_CHUNK_SIZE)  # Split the document into smaller chunks
    
    # Summarize chunks in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        summarized_chunks = list(executor.map(summarize_chunk, chunks))
    
    summarized_chunks = [clean_output(summary) for summary in summarized_chunks if summary]
    
    # Combine all summarized chunks into a single summary
    return "\n\n".join(summarized_chunks)

# Function to summarize the entire document (for DOCX files)
def summarize_entire_document_docx(paragraphs):
    chunks = split_into_chunks(paragraphs, max_chars=DOCX_CHUNK_SIZE)  # Split the document into larger chunks
    
    summarized_chunks = []
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        if summary:
            cleaned_summary = clean_output(summary)
            summarized_chunks.append(cleaned_summary)
    
    return "\n\n".join(summarized_chunks)

# Function to create a .docx file with the summarized text
def create_summary_docx(summary):
    doc = Document()
    
    # Add the summary as the main content
    doc.add_heading('Résumé du cours de droit', 0)
    doc.add_paragraph(summary)
    
    # Save the document to a buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to create a PDF file with the summarized text
def create_summary_pdf(summary):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add smaller font size and tighter line spacing
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=10)  # Reduced font size for more condensed output

    clean_summary = summary.replace("\n\n", "\n")  # Proper paragraph breaks but not double-spaced
    pdf.multi_cell(0, 5, clean_summary)  # Adjust cell height (line spacing) to 5

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Streamlit interface
st.title("Résumé de cours")

# File uploader with a unique key
uploaded_file = st.file_uploader(
    "Téléchargez un fichier de cours (.docx ou .pdf)", 
    type=["docx", "pdf"],
    key="file_uploader_1"
)

# Initialize variables for paragraphs and summary
paragraphs = None
summary = None

# Process file after it is uploaded
if uploaded_file is not None:
    # Detect file type and read accordingly
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        paragraphs = read_docx(uploaded_file)
        output_format = "docx"
    elif uploaded_file.type == "application/pdf":
        paragraphs = read_pdf(uploaded_file)
        output_format = "pdf"

    st.write("Aperçu des 5 premiers paragraphes :")
    st.text_area("Texte du document", "\n\n".join(paragraphs[:5]), height=300)

# Summarization button
if st.button("Générer un résumé"):
    with st.spinner("Génération du résumé..."):
        try:
            if output_format == "pdf":
                summary = summarize_entire_document_pdf(paragraphs)  # Optimized for PDF
            else:
                summary = summarize_entire_document_docx(paragraphs)  # Regular for DOCX
            if summary:
                st.success("Résumé généré avec succès !")
                st.write(summary)
        except Exception as e:
            st.error(f"Une erreur s'est produite pendant la génération du résumé : {e}")

# Only allow download if summarization was successful
if summary:
    if output_format == "pdf":
        pdf_file = create_summary_pdf(summary)
        st.download_button(
            label="Télécharger le résumé en .pdf",
            data=pdf_file,
            file_name="petit_resume_cours.pdf",
            mime="application/pdf"
        )
    else:
        docx_file = create_summary_docx(summary)
        st.download_button(
            label="Télécharger le résumé en .docx",
            data=docx_file,
            file_name="petit_resume_cours.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
