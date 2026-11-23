import os
import chromadb
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# 1. PDF FOLDER
# ==========================================

PDF_FOLDER = "documents"


# ==========================================
# 2. READ PDF
# ==========================================

documents = []
ids = []

for filename in os.listdir(PDF_FOLDER):

    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, filename)

        print("Reading:", filename)

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


        # ==========================================
        # 3. SPLIT TEXT INTO CHUNKS
        # ==========================================

        chunk_size = 1500

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size].strip()

            if chunk:

                documents.append(chunk)

                ids.append(
                    f"{filename}_{i}"
                )


# ==========================================
# 4. CHECK PDF
# ==========================================

if not documents:

    print("❌ No PDF found in documents folder.")
    exit()


print("Total chunks:", len(documents))


# ==========================================
# 5. CREATE TF-IDF VECTORS
# ==========================================

print("Creating vectors...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

vectors = vectorizer.fit_transform(documents)


# Convert sparse matrix to normal lists
embeddings = vectors.toarray().tolist()


# ==========================================
# 6. CREATE CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="vector_db"
)


# Delete old collection if it exists
try:

    client.delete_collection(
        name="sports_documents"
    )

except Exception:

    pass


collection = client.create_collection(
    name="sports_documents"
)


# ==========================================
# 7. STORE DOCUMENTS + VECTORS
# ==========================================

collection.add(

    documents=documents,

    embeddings=embeddings,

    ids=ids
)


# ==========================================
# 8. SAVE VECTOR DB
# ==========================================

print()
print("======================================")
print("✅ VECTOR DATABASE CREATED")
print("======================================")

print("PDF chunks:", len(documents))

print("Database folder: vector_db")

print("Collection: sports_documents")