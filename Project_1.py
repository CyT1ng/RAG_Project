'''
Project 1: PDF semantic search (RAG IMPLEMENTATION)
1. Download the pdf: https://s201.q4cdn.com/141608511/files/doc_financials/2025/q2/78501ce3-7816-4c4d-8688-53dd140df456.pdf 
2. Use unstructured lib to convert pdf into texts, page by page
3. Split the full text into smaller chunks of paragraphs, each paragraph only contain 300-500 tokens, using “tiktoken” lib
4. For each chunk use openai "text-embedding-small” to convert to embedding
5. store into spreadsheet / Vector DB such as chroma, with header “text”, “embedding”,”n_tokens”
6. write a function to get the top K most similar paragraphs given the user question using embedding search
7. Filter out the paragraphs < threshold (0.7)
8. Use LLM to answer the question based on the top K most similar paragraphs.
9. Try 10 different questions to evaluate the quality of RAG
'''
import os
from unstructured.partition.pdf import partition_pdf
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import numpy as np

LOCAL_PDF_PATH = "/Users/j.c./Documents/PythonProjects/GenAI_Learning/Project_1/nvidia_q2_2025.pdf"
MIN_TOKEN = 300
MAX_TOKEN = 500
TOP_K = 10 # Number of top similar chunks to return.
THRESHOLD = 0.5

CSV_FILEPATH = "nvidia_q2_2025_embeddings.csv"

CHROMA_DIR = "./chroma_db"

# Load environment variables from .env file.
load_dotenv() # Load the API key from the .env file.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

def file_to_pages() -> list[str]:
    elements = partition_pdf(
        filename=LOCAL_PDF_PATH,
        strategy="fast",
        languages=["eng"]
    )
    pages = []
    for element in elements:
        pages.append(getattr(element, "text", ""))
    print(f"Processed {len(pages)} pages")
    return pages

def pages_to_paragraphs(pages: list[str]) -> list[str]:
    paragraphs = []
    for page in pages:
        # Split by blank lines, trim whitespace, and keep only non-empty blocks.
        paragraph_blocks = [block.strip() for block in page.split("\n\n") if block.strip()]
        for paragraph in paragraph_blocks:
            paragraphs.append(paragraph)
    print(f"Extracted {len(paragraphs)} paragraphs")
    return paragraphs

#paragraphs to sentences

def paragraphs_to_chunks(paragraphs: list[str]) -> list[str]:
    chunks = []
    idx = 0
    n = len(paragraphs)
    while idx < n:
        currentPara = paragraphs[idx]
        # Count current paragraph's tokens using tiktoken.
        # For tiktoken, choose an encoding consistent with OpenAI embeddings model; many use "cl100k_base".
        tokenNum = len(tiktoken.get_encoding("cl100k_base").encode(currentPara))
        idx2 = idx + 1
        # Keep adding paragraphs until reaching at least MIN_TOKEN.
        while tokenNum < MIN_TOKEN and idx2 < n:
            currentPara += "\n\n" + paragraphs[idx2]
            tokenNum = len(tiktoken.get_encoding("cl100k_base").encode(currentPara))
            idx2 += 1
            if tokenNum > MAX_TOKEN:
                break
        chunks.append(currentPara)
        idx = idx2
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(model="text-embedding-3-small", input=chunk)
        embeddings.append(response.data[0].embedding)
    print(f"Generated embeddings for {len(chunks)} chunks")
    return embeddings

def storeInChromaDB(chunks: list[str], embeddings: list[list[float]]):
    client_db = chromadb.PersistentClient(path=CHROMA_DIR) # not default to cos similarity
    collection = client_db.get_or_create_collection(name="nvda_chunks")
    # Prepare ids, metadatas, documents
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = chunks
    metadatas = [{"n_tokens": len(tiktoken.get_encoding("cl100k_base").encode(chunk))} for chunk in chunks]
    # Upsert = update or insert the data into the collection.
    collection.upsert(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
    print(f"Upserted {len(documents)} documents into Chroma collection.")

def cosineSimilarity(vec_1, vec_2) -> float:
    dot_product = np.dot(vec_1, vec_2)
    magnitude_1 = np.linalg.norm(vec_1)
    magnitude_2 = np.linalg.norm(vec_2)
    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0.0
    return dot_product / (magnitude_1 * magnitude_2)

def get_top_k_similar_paragraphs(query: str, embeddings: list[list[float]], chunks: list[str]) -> list[tuple[str, float]]:
    # Generate embedding for the query
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_embedding = response.data[0].embedding

    similarities = []
    for i, embedding in enumerate(embeddings):
        similarity = cosineSimilarity(query_embedding, embedding)
        if similarity >= THRESHOLD:
            similarities.append((i, similarity))

    # Sort by similarity score in descending order and select top K
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:TOP_K]

    # Return the corresponding paragraphs
    return [(chunks[i], similarity) for i, similarity in top_k]

def llmAnswer(query: str, top_k_paragraphs: list[tuple[str, float]]) -> str:
    context = "\n".join([f"Paragraph {i+1}: {text}" for i, (text, _) in enumerate(top_k_paragraphs)])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Only use information from the context to answer the question."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    pages = file_to_pages()
    paragraphs = pages_to_paragraphs(pages)
    chunks = paragraphs_to_chunks(paragraphs)
    embeddings = embed_chunks(chunks)
    #saveInCSV(chunks, embeddings, CSV_FILEPATH)
    if chunks and embeddings and len(chunks) == len(embeddings):
        storeInChromaDB(chunks, embeddings)
    
    querys =  [
        "What were NVIDIA's earnings for Q2 2025?",
        "How did NVIDIA's revenue change compared to Q2 2024?",
        "What is NVIDIA's outlook for the next quarter?",
        "Did NVIDIA announce any new products?",
        "What is NVIDIA's strategy for AI development?",
        "How is NVIDIA addressing supply chain challenges?",
        "What are the key risks mentioned in the report?",
        "How does NVIDIA's performance compare to competitors?",
        "What is NVIDIA's approach to sustainability?",
        "What are the highlights from NVIDIA's shareholder meeting?"
    ]
    for idx, q in enumerate(querys, 1):
        print("="*150)
        print(f"Q{idx}: {q}")
        top_k_paragraphs = get_top_k_similar_paragraphs(q, embeddings, chunks) # Debug: print top_k_paragraphs create a csv
        answer = llmAnswer(q, top_k_paragraphs)
        print(f"Answer: {answer}")