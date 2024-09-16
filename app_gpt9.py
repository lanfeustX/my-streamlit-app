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
PDF_CHUNK_SIZE = 3000  # Smaller chunk size for faster PDF processing
DOCX_CHUNK_SIZE = 6000  # Larger chunk size for DOCX files
MAX_WORKERS = 4  # Number of threads for parallel processing
MAX_CHUNK_TOKENS = 4000  # Maximum token limit per chunk (for safety)
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

# Function to extract a table of contents (TOC) based on document structure
def extract_toc(paragraphs):
    toc = []
    for i, para in enumerate(paragraphs):
        # Identify chapter, section, or subsection based on common patterns
        if re.match(r'(Chapitre|Section|Partie|Sous-section|Article)\s\d+', para):
            toc.append(f"{para} (Page {i+1})")
    return toc

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

# Function to summarize a chunk of text, ensuring it doesn't exceed token limits
def summarize_chunk(chunk):
    prompt = (
        f"Vous êtes un assistant pédagogique chargé de condenser un cours de droit. "
        f"Faites un résumé concis et fluide en regroupant les idées dans des paragraphes cohérents. "
        f"Conservez uniquement les points clés essentiels et faites un résumé exhaustif du texte suivant :\n\n"
        f"Texte à résumer :\n\n{chunk}"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1800,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")
        return None

# Function to clean and format the output text
def clean_output(text):
    cleaned_text = re.sub(r'\bRésumé[: ]*', '', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to summarize the entire document (for PDFs) with parallel processing
def summarize_entire_document_pdf(paragraphs):
    paragraphs = preprocess_paragraphs(paragraphs)
    chunks = split_into_chunks(paragraphs, max_chars=PDF_CHUNK_SIZE)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        summarized_chunks = list(executor.map(summarize_chunk, chunks))
    
    summarized_chunks = [clean_output(summary) for summary in summarized_chunks if summary]
    
    return "\n\n".join(summarized_chunks)

# Function to summarize the entire document (for DOCX files)
def summarize_entire_document_docx(paragraphs):
    chunks = split_into_chunks(paragraphs, max_chars=DOCX_CHUNK_SIZE)
    
    summarized_chunks = []
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        if summary:
            cleaned_summary = clean_output(summary)
            summarized_chunks.append(cleaned_summary)
    
    return "\n\n".join(summarized_chunks)

# Function to create a .docx file with the summarized text or table of contents
def create_docx_file(content, heading):
    doc = Document()
    doc.add_heading(heading, 0)
    doc.add_paragraph(content)
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to create a PDF file with the summarized text or table of contents
def create_pdf_file(content):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(0, 5, content)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Streamlit interface
st.title("Résumé de cours ou Table des matières")

# File uploader
uploaded_file = st.file_uploader("Téléchargez un fichier de cours (.docx ou .pdf)", type=["docx", "pdf"])

# Initialize variables for paragraphs, summary, and table of contents
paragraphs = None
output = None

# Checkbox to select between generating a résumé or table of contents
generate_toc = st.checkbox("Générer une table des matières")

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

# Button to trigger summarization or table of contents generation
if st.button("Générer"):
    with st.spinner("Génération en cours..."):
        try:
            if generate_toc:
                toc = extract_toc(paragraphs)
                output = "\n".join(toc)
            else:
                if output_format == "pdf":
                    output = summarize_entire_document_pdf(paragraphs)
                else:
                    output = summarize_entire_document_docx(paragraphs)

            if output:
                st.success("Génération réussie !")
                st.write(output)

        except Exception as e:
            st.error(f"Une erreur s'est produite pendant la génération : {e}")

# Allow download of the generated document
if output:
    if generate_toc:
        heading = "Table des matières"
    else:
        heading = "Résumé du cours"

    if output_format == "pdf":
        pdf_file = create_pdf_file(output)
        st.download_button(
            label=f"Télécharger le {heading} en .pdf",
            data=pdf_file,
            file_name=f"{heading}.pdf",
            mime="application/pdf"
        )
    else:
        docx_file = create_docx_file(output, heading)
        st.download_button(
            label=f"Télécharger le {heading} en .docx",
            data=docx_file,
            file_name=f"{heading}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
