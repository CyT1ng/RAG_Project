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

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import uuid
import os


file_path = "/Users/j.c./Documents/PythonProjects/GenAI_Learning/Project_1/nvidia_q2_2025.pdf"
top_k = 10 # Number of top similar chunks to return.
THRESHOLD = 0.2

chroma_db_path = "./chroma_db"

# Load environment variables from .env file.
load_dotenv() 
# Load the API key from the .env file.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

def file_to_pages() -> List[Dict]:
    pages = []
    elements = partition_pdf(filename=file_path,strategy="fast",languages=["eng"])
    page_texts = {}
    for element in elements:
        # Get the page number; default to 1 if not provided
        page_number = getattr(element.metadata, "page_number", None)
        page_number = int(page_number) if page_number is not None else 1
        # Append element text to the corresponding page's list
        page_texts.setdefault(page_number, []).append(str(element))

    for page_number, texts in page_texts.items():
        combined_text = " ".join(texts).strip()
        pages.append({
            "text": combined_text,
            "source": str(file_path),
            "page_number": page_number,
        })
    print(f"Processed {len(pages)} pages")
    return pages

# Problem: chunks may divide a sentence in the middle.
def pages_to_chunks(pages: List[Dict]) -> List[Dict]:
    class ChunkingStrategy:
        def __init__(self, method: str = 'recursive', encoding_name: str = "cl100k_base", 
                    chunk_size: int = 300, chunk_overlap: int = 50):
            """
            Initialize a chunking strategy.

            Args:
                method (str): The chunking method, e.g. 'fixed' or 'recursive'.
                encoding_name (str): The name of the encoding to use.
                chunk_size (int): The size of each chunk.
                chunk_overlap (int): The overlap between chunks.
            """
            self.method = method
            self.encoding_name = encoding_name
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def chunk_document(self, text: str) -> List[str]:
            # Use Recursive Method to chunk the text.
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.encoding_name, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_text(text)

    chunker = ChunkingStrategy(chunk_size=300, chunk_overlap=50)
    all_chunks = []
    for page in pages:
        chunks = chunker.chunk_document(page["text"])
        for chunk in chunks:
            all_chunks.append({
            "id": str(uuid.uuid4()),  # Generate a unique ID for each chunk
            "text": chunk,
            "metadata": {
                "source": page["source"],
                "page_number": page["page_number"],
            }
        })
    print(f"Created {len(all_chunks)} chunks")
    return all_chunks

def embed_chunks(chunks: List[Dict]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(model="text-embedding-3-small", input=chunk["text"])
        embeddings.append(response.data[0].embedding)
    print(f"Generated embeddings for {len(chunks)} chunks")
    return embeddings

def storeInChromaDB(chunks: List[str], embeddings: List[List[float]]):
    client_db = chromadb.PersistentClient(path=chroma_db_path)
    # Using cosine distance instead of euclidean distance.
    collection = client_db.get_or_create_collection(
        name="nvda_embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks]
        )
    print("Embeddings stored in ChromaDB.")
'''
def get_top_k_similar_chunks(query: str) -> List[Dict]:
    # Generate embedding for the query
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_embedding = response.data[0].embedding
    # Problem: Not default to cos similarity.
    client_db = chromadb.PersistentClient(path=chroma_db_path) 
    collection = client_db.get_collection(name="nvda_embeddings")
    # Get the top K similar chunks
    top_k_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
        )

    reformatted = []

    # Get the lists from the results. They are expected to be lists of lists.
    metadatas = top_k_results.get("metadatas", [])
    documents = top_k_results.get("documents", [])
    distances = top_k_results.get("distances", [])
    
    # Loop over each group (each inner list represents one set of matches)
    chunk_index = 1
    for meta_group, doc_group, distance_group in zip(metadatas, documents, distances):
        # Iterate over each item in the inner lists
        for meta, text, distance in zip(meta_group, doc_group, distance_group):
            item = {
                "chunk_index": chunk_index,
                "page_number": meta.get("page_number"),
                "source": meta.get("source"),
                "text": text,
                "distance": distance,
                "score": 1 - distance
            }
            reformatted.append(item)
            chunk_index += 1
    return reformatted
'''
def get_top_k_similar_chunks(query: str) -> List[Dict]:
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_embedding = response.data[0].embedding

    client_db = chromadb.PersistentClient(path=chroma_db_path)
    collection = client_db.get_collection(name="nvda_embeddings")

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 4) Convert distances -> cosine similarity and reformat
    #    For cosine space: similarity = 1 - cosine_distance
    results = []
    chunk_index = 1

    metadatas_groups  = res.get("metadatas", [])
    documents_groups  = res.get("documents", [])
    distances_groups  = res.get("distances", [])

    for metas, docs, dists in zip(metadatas_groups, documents_groups, distances_groups):
        for meta, text, dist in zip(metas, docs, dists):
            cosine_sim = 1.0 - dist
            results.append({
                "chunk_index": chunk_index,
                "page_number": meta.get("page_number"),
                "source": meta.get("source"),
                "text": text,
                "distance": dist,          # cosine distance
                "similarity": cosine_sim,  # cosine similarity
            })
            chunk_index += 1

    if results:
        print(f"Similarity scores range: {min(r['similarity'] for r in results):.3f} to {max(r['similarity'] for r in results):.3f}")
    
    results = [r for r in results if r["similarity"] >= THRESHOLD]

    # 6) Sort by similarity (descending) to be explicit
    results.sort(key=lambda r: r["similarity"], reverse=True)

    return results

    
def llmAnswer(query: str, top_k_chunks: List[Dict]) -> str:
    # Merge all text values from the list of dictionaries
    context = "\n".join([f"Chunk {chunk['chunk_index']}: {chunk['text']}" for chunk in top_k_chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Only use information from the context to answer the question."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def run_embedding_pipeline():
    """
    Pipeline 1: Process PDF, create chunks, generate embeddings, and store in ChromaDB
    """
    print("="*50)
    
    pages = file_to_pages()
    chunks = pages_to_chunks(pages)
    embeddings = embed_chunks(chunks)
    storeInChromaDB(chunks, embeddings)

    print("Embedding pipeline completed successfully!")
    print("Data is now stored in ChromaDB and ready for querying.")

    return chunks, embeddings

def run_query_pipeline():
    """Pipeline 2: Query the ChromaDB and generate answers using LLM"""
    print("="*50)
    
    # Load data from ChromaDB
    client_db = chromadb.PersistentClient(path=chroma_db_path)
    collection = client_db.get_collection(name="nvda_embeddings")
    
    # Get all documents from the collection
    results = collection.get()
    chunks = results['documents']
    
    print(f"Loaded {len(chunks)} chunks from ChromaDB")
        
    # Test questions
    
    queries = [
        "What is NVIDIA's strategy for AI development?",
        "Describe NVIDIA's recent advancements in GPU technology.",
        "How is NVIDIA leveraging partnerships to expand its market presence?",       
        "Summarize the key challenges faced by NVIDIA in the last fiscal year.",
        "How does NVIDIA plan to address future market uncertainties?",
        "What are the most significant opportunities for NVIDIA in the AI sector?",
        "How has NVIDIA's revenue evolved over the past year?",
        "What are the main takeaways from NVIDIA's management discussion and analysis?",
    ]

    '''

    queries = str(input("Enter your question: "))

    '''

    print(f"\nTesting {len(queries)} queries...")
    
    for idx, q in enumerate(queries, 1):
        print("="*100)
        print(f"Q{idx}: {q}")
        # Need a Debug: print top_k_paragraphs create a csv.
        top_k_chunks = get_top_k_similar_chunks(q) 
        
        # Debug: Print information about the results
        print(f"Found {len(top_k_chunks)} chunks above threshold {THRESHOLD}")
        
        if len(top_k_chunks) == 0:
            print("No chunks found above the similarity threshold.")
            continue
        
        answer = llmAnswer(q, top_k_chunks)
        print(f"Answer: {answer}")
        '''
        print(f"\n=== DETAILED CHUNK INFORMATION ===")
        for chunk in top_k_chunks:
            print(f"\nRelated Chunk {chunk['chunk_index']}:\n{chunk['text']}")
            print(f"Page: {chunk['page_number']}")
            print(f"Source: {chunk['source']}")
            print(f"Distance: {chunk['distance']}")
            print(f"Score: {chunk['similarity']}")
            print("-"*100)
        '''

def show_interactive_menu():
    """Show an interactive menu to choose which pipeline to run"""
    while True:   
        choice = input("Enter your choice (embed/query/both/quit): ").strip().lower()
        
        if choice == "embed":
            print("\nRunning embedding pipeline...")
            run_embedding_pipeline()
            
        elif choice == "query":
            print("\nRunning query pipeline...")
            run_query_pipeline()
            
        elif choice == "both":
            print("\nRunning both pipelines...")
            chunks, embeddings = run_embedding_pipeline()
            if chunks and embeddings:
                run_query_pipeline()
                
        elif choice == "quit":
            print("Program terminated.")
            break
            
        else:
            print(f"Invalid choice: '{choice}'. Please try again.")
        
        # Ask if user wants to continue
        if choice != "quit":
            continue_choice = input("\nDo you want to run another pipeline? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("Program terminated.")
                break

if __name__ == "__main__":
    show_interactive_menu()