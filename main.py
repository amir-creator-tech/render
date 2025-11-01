# main.py
import os
import io
import re
import json
from typing import List, Tuple
from flask import Flask, request, render_template_string, send_file, jsonify
from werkzeug.utils import secure_filename
import pdfplumber
import docx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import openai  # optional: used only if OPENAI_API_KEY provided

# ---- Setup ----
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

OPENAI_KEY = os.getenv('OPENAI_API_KEY')  # optional
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit for uploads

# ---- Utilities: text extraction ----
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF content (bytes)."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """Extract text from DOCX content (bytes)."""
    doc = docx.Document(io.BytesIO(docx_bytes))
    paras = [p.text for p in doc.paragraphs if p.text and not p.text.isspace()]
    return "\n\n".join(paras)

# ---- Text processing utilities ----
def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def paragraph_tokenize(text: str) -> List[str]:
    # split on two newlines or long line breaks
    paras = [clean_text(p) for p in re.split(r'\n\s*\n', text) if len(clean_text(p))>30]
    return paras

def chunk_sentences(paragraphs: List[str]) -> List[str]:
    """Break paragraphs into smaller sentence chunks for better topic clustering."""
    chunks = []
    for p in paragraphs:
        sents = sent_tokenize(p)
        # group small groups of sentences (2-4) as a chunk
        n = len(sents)
        if n == 0:
            continue
        i = 0
        while i < n:
            chunk = " ".join(sents[i:i+3])
            chunks.append(clean_text(chunk))
            i += 3
    return chunks

# ---- Topic segmentation ----
def topics_from_text(text: str, n_topics: int = 6) -> List[Tuple[str, List[str]]]:
    """
    Returns a list of (topic_label, list_of_sentences) tuples.
    Uses TF-IDF + KMeans clustering on sentence chunks.
    """
    paragraphs = paragraph_tokenize(text)
    chunks = chunk_sentences(paragraphs)
    if len(chunks) < n_topics:
        n_topics = max(1, len(chunks))
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english')
    X = vectorizer.fit_transform(chunks)
    if len(chunks) >= n_topics and n_topics > 1:
        km = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        topics = []
        for i in range(n_topics):
            top_terms = [terms[ind] for ind in order_centroids[i, :6]]
            # sentences in this cluster
            sents = [chunks[j] for j in range(len(chunks)) if labels[j] == i]
            topics.append((", ".join(top_terms[:4]), sents))
    else:
        # fallback: single topic
        topics = [("general", chunks)]
    return topics

# ---- Helper: detect example/experiment/construction sentences ----
KEYWORDS_EXAMPLE = ['experiment','observe','measurement','measure','setup','procedure','result','sample','method','apparatus']
KEYWORDS_CONSTRUCTION = ['assemble','construction','construct','build','wiring','setup','connect']
def extract_section_sentences(sentences: List[str], keywords: List[str], max_items=3) -> List[str]:
    hits = []
    for s in sentences:
        s_l = s.lower()
        if any(k in s_l for k in keywords):
            hits.append(s)
            if len(hits) >= max_items:
                break
    return hits

# ---- MCQ generation (local heuristic or optional OpenAI) ----
def generate_mcqs_local(sentences: List[str], num_mcq=3):
    """Simple heuristic to create MCQs by masking nouns/keywords in sentences."""
    mcqs = []
    for s in sentences:
        words = [w for w in word_tokenize(s) if re.match(r'^\w+$', w)]
        # pick a candidate word that is not a stopword and reasonably long
        candidate = None
        for w in words[::-1]:
            if w.lower() not in STOPWORDS and len(w) > 3:
                candidate = w
                break
        if not candidate:
            continue
        question = s.replace(candidate, "_____")
        # build simple distractors by small edits
        opts = [candidate, candidate + 's', candidate[::-1], candidate + '1']
        # ensure uniqueness and shuffle
        opts = list(dict.fromkeys(opts))[:4]
        mcqs.append({'q': question, 'options': opts, 'answer': opts[0]})
        if len(mcqs) >= num_mcq:
            break
    return mcqs

def generate_mcqs_openai(prompt_text: str, num=3):
    """Use OpenAI to generate MCQs; fallback to local if no key or failure."""
    if not OPENAI_KEY:
        return []
    try:
        # simple prompt: ask for JSON with mcqs
        prompt = f"From the following text, create {num} multiple choice questions (JSON list with q, options (4), answer):\n\n{prompt_text}\n\nReturn JSON only."
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # example model name — change per availability
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=500
        )
        content = resp['choices'][0]['message']['content']
        # try parse JSON
        data = json.loads(content)
        return data
    except Exception as e:
        print("OpenAI MCQ generation failed:", e)
        return []

# ---- LaTeX slide generator ----
def make_latex_slide(topic_title: str, sentences: List[str], mcqs: List[dict], example_lines: List[str], construct_lines: List[str]) -> str:
    """Return a LaTeX frame (beamer slide) for a topic."""
    safe_title = re.sub(r'[_&%$#{}]', '', topic_title)[:80]
    slide = []
    slide.append("\\begin{frame}{%s}\n" % safe_title)
    # main bullet points (extract top 4 sentences)
    slide.append("\\textbf{Key points:}\n\\begin{itemize}")
    for s in sentences[:4]:
        slide.append("\\item %s\n" % s.replace('%','\\%'))
    slide.append("\\end{itemize}\n")
    # example/experiment
    if example_lines:
        slide.append("\\textbf{Example / Experiment:}\n\\begin{itemize}")
        for e in example_lines:
            slide.append("\\item %s\n" % e.replace('%','\\%'))
        slide.append("\\end{itemize}\n")
    # construction/measurement
    if construct_lines:
        slide.append("\\textbf{Constructions \\& Measurement:}\n\\begin{itemize}")
        for c in construct_lines:
            slide.append("\\item %s\n" % c.replace('%','\\%'))
        slide.append("\\end{itemize}\n")
    # worksheet (MCQs)
    if mcqs:
        slide.append("\\textbf{Worksheet (MCQs):}\n")
        for i, m in enumerate(mcqs):
            slide.append("\\textbf{Q%d.} %s\\\\" % (i+1, m['q'].replace('%','\\%')))
            for j,opt in enumerate(m['options']):
                slide.append("\\quad (%c) %s\\\\" % (chr(97+j), opt.replace('%','\\%')))
            slide.append("\\vspace{2mm}\n")
    slide.append("\\end{frame}\n\n")
    return "\n".join(slide)

def assemble_full_latex(slides: List[str], doc_title: str = "Generated Slides") -> str:
    header = r"""\documentclass{beamer}
\usetheme{Madrid}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\begin{document}
\title{%s}
\author{Auto-generated}
\date{\today}
\frame{\titlepage}
""" % doc_title
    footer = "\\end{document}\n"
    body = "\n".join(slides)
    return header + body + footer

# ---- Flask routes ----
INDEX_HTML = """
<!doctype html>
<title>SlideMaker</title>
<h2>Upload PDF and DOCX to generate LaTeX slides</h2>
<form method=post enctype=multipart/form-data action="/process">
  PDF: <input type=file name=pdf required><br><br>
  DOCX (or .txt): <input type=file name=doc required><br><br>
  Title: <input type=text name=title value="My Slides"><br><br>
  <input type=submit value=Upload>
</form>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/process', methods=['POST'])
def process():
    # receive files
    pdf_file = request.files.get('pdf')
    doc_file = request.files.get('doc')
    title = request.form.get('title','Generated Slides')
    if not pdf_file or not doc_file:
        return "Both files required (pdf and docx/txt).", 400
    pdf_bytes = pdf_file.read()
    doc_bytes = doc_file.read()

    # extract text
    text_pdf = extract_text_from_pdf_bytes(pdf_bytes)
    # try docx; fallback to plain text decode
    try:
        text_doc = extract_text_from_docx_bytes(doc_bytes)
    except Exception:
        try:
            text_doc = doc_bytes.decode('utf-8')
        except:
            text_doc = ""

    full_text = (text_pdf + "\n\n" + text_doc).strip()
    if len(full_text) < 50:
        return "Could not extract meaningful text from uploads.", 400

    # topic segmentation
    topics = topics_from_text(full_text, n_topics=6)

    # for each topic build slide
    slides = []
    for topic_label, sents in topics:
        # detect examples & constructions heuristically
        example_lines = extract_section_sentences(sents, KEYWORDS_EXAMPLE, max_items=2)
        construct_lines = extract_section_sentences(sents, KEYWORDS_CONSTRUCTION, max_items=2)

        # generate MCQs: try OpenAI if key exists else local heuristic
        mcqs = []
        if OPENAI_KEY:
            mcqs = generate_mcqs_openai(" ".join(sents[:6]), num=3)
            if not mcqs:
                mcqs = generate_mcqs_local(sents, num_mcq=3)
        else:
            mcqs = generate_mcqs_local(sents, num_mcq=3)

        slide_tex = make_latex_slide(topic_label, sents, mcqs, example_lines, construct_lines)
        slides.append(slide_tex)

    full_tex = assemble_full_latex(slides, doc_title=title)

    # return as downloadable file
    out_name = secure_filename(title.replace(" ", "_") + ".tex")
    return send_file(io.BytesIO(full_tex.encode('utf-8')), attachment_filename=out_name, as_attachment=True, mimetype='application/x-tex')

# Run the app (for local dev)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
