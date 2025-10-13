'''
Basic RAG:
1. Download the file (In this case, a pdf)
2. Use unstructured lib to convert pdf into texts, page by page
3. Split the full text into smaller chunks of paragraphs, each paragraph only contain 300-500 tokens, using “tiktoken” lib
4. For each chunk use openai "text-embedding-small” to convert to embedding
5. store into spreadsheet / Vector DB such as chroma, with header “text”, “embedding”,”n_tokens”
6. write a function to get the top K most similar paragraphs given the user question using embedding search
7. Filter out the paragraphs < threshold (0.5)
8. Use LLM to answer the question based on the top K most similar paragraphs.
9. Try 10 different questions to evaluate the quality of RAG

Hypothetical Document Embeddings Enhanced RAG:
1. In the data ingestion pipeline:
    - Generate multiple but not identical hypothetical documents for every chunk.
    - Each hypothetical question will have a metadata field that indicates the corresponding chunk_id.
    - Generate embeddings for hypothetical questions.
    - Store hypothetical questions and chunks in ChromaDB with 2 separate collections: chunks-collection and hypothetical-collection.
2. In the query pipeline:
    - When retrieving top-k similar chunks, concurrently retrieve top-k similar hypothetical questions and their corresponding chunks.
    - Aggregate the chunks and hypothetical questions related chunks and send them to LLM.
'''

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from html.parser import HTMLParser
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import uuid
import os


file_path = "/Users/j.c./Documents/PythonProjects/GenAI_Learning/Project_1/nvidia_q2_2025.pdf"
top_k = 5 # Number of top similar chunks to return.
THRESHOLD = 0.5

chroma_db_path = "./chroma_db"

# Load environment variables from .env file.
load_dotenv() 
# Load the API key from the .env file.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

class _HTMLTableParser(HTMLParser):
    """Extract rows from HTML table fragments emitted by unstructured."""

    def __init__(self) -> None:
        super().__init__()
        self._tables: List[List[List[str]]] = []
        self._current_rows: List[List[str]] = []
        self._current_cells: List[str] = []
        self._capture_cell = False

    def handle_starttag(self, tag: str, attrs):
        if tag == "table":
            self._current_rows = []
        elif tag == "tr":
            self._current_cells = []
        elif tag in {"td", "th"}:
            self._capture_cell = True
            self._current_cell_data: List[str] = []

    def handle_data(self, data: str):
        if self._capture_cell:
            self._current_cell_data.append(data)

    def handle_endtag(self, tag: str):
        if tag in {"td", "th"} and self._capture_cell:
            cell_text = "".join(self._current_cell_data).strip()
            self._current_cells.append(cell_text)
            self._capture_cell = False
        elif tag == "tr":
            if self._current_cells:
                self._current_rows.append(self._current_cells)
            self._current_cells = []
        elif tag == "table":
            if self._current_rows:
                self._tables.append(self._current_rows)
            self._current_rows = []

    def tables(self) -> List[List[List[str]]]:
        return self._tables


def _html_table_to_markdown(html: str) -> str:
    parser = _HTMLTableParser()
    parser.feed(html)
    parser.close()

    tables = parser.tables()
    markdown_blocks: List[str] = []
    for table in tables:
        markdown = _table_to_markdown(table)
        if markdown:
            markdown_blocks.append(markdown)
    return "\n\n".join(markdown_blocks)


def _table_to_markdown(table: List[List[str]]) -> str:
    if not table:
        return ""

    column_count = max((len(row or []) for row in table), default=0)
    if column_count == 0:
        return ""

    def _sanitize(row: List[str]) -> List[str]:
        row = row or []
        cells = [(cell or "").strip() for cell in row]
        if len(cells) < column_count:
            cells.extend([""] * (column_count - len(cells)))
        return cells

    rows = [_sanitize(row) for row in table]
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []

    markdown_lines = ["| " + " | ".join(header) + " |"]
    markdown_lines.append("| " + " | ".join("---" for _ in range(column_count)) + " |")
    for body_row in body:
        markdown_lines.append("| " + " | ".join(body_row) + " |")

    return "Table:\n" + "\n".join(markdown_lines)


def _extract_pdf_tables(pdf_path: str) -> Dict[int, List[str]]:
    tables_by_page: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                raw_tables = page.extract_tables() or []
                markdown_tables = []
                for raw_table in raw_tables:
                    markdown = _table_to_markdown(raw_table)
                    if markdown:
                        markdown_tables.append(markdown)
                if markdown_tables:
                    tables_by_page[index] = markdown_tables
    except Exception as exc:
        print(f"Warning: pdfplumber failed to extract tables: {exc}")
    return tables_by_page


def _element_to_markdown(element) -> str:
    category = getattr(element, "category", "") or getattr(element, "type", "")
    if isinstance(category, str) and category.lower() == "table":
        html = getattr(getattr(element, "metadata", None), "text_as_html", None)
        if html:
            markdown = _html_table_to_markdown(html)
            if markdown:
                return markdown
    return str(element).strip()


def file_to_pages() -> List[Dict]:
    pages = []
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        include_tables=True,
        languages=["eng"])
    fallback_tables = _extract_pdf_tables(file_path)
    page_blocks: Dict[int, List[str]] = {}

    for element in elements:
        page_number = getattr(element.metadata, "page_number", None)
        page_number = int(page_number) if page_number is not None else 1
        text_block = _element_to_markdown(element)
        if not text_block:
            continue
        page_blocks.setdefault(page_number, []).append(text_block)

    for page_number, tables in fallback_tables.items():
        page_blocks.setdefault(page_number, []).extend(tables)

    for page_number, blocks in page_blocks.items():
        combined_text = "\n\n".join(blocks).strip()
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

def generate_hypothetical_questions(chunks: List[Dict]) -> List[Dict]:
    hypothetical_questions = []
    for chunk in chunks:
        markdown_prompt = """
            #Task:
            - You are an expert in text analysis, generate multiple hypothetical questions for a given chunk.
            #Constraints
            - Don't generate the same or highly similar question.
            - Generate at least 2 hypothetical question.
            - Don't generate more than 5 hypothetical questions.
            #Output
            - A python list of hypothetical questions separated by semicolons.
            #Example
            - Chunk: Nvidia's revenue for Q2 2025 was $1.2 billion, up from $1.1 billion in Q2 2024.
            - Output: "What is Nvidia's revenue for Q2 2025?; What is Nvidia's revenue for Q2 2024?; How did Nvidia's revenue change compared to Q2 2024?"
        """
        
        hypothetical_question = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": markdown_prompt
                },
                {
                    "role": "user",
                    "content": chunk["text"]
                }
            ]
        )
        output_text = getattr(hypothetical_question, "output_text", "")
        hyperthetical_question_list = [q.strip() for q in output_text.split(";") if q.strip()]
        # LLM output needs to be parsed into a list of strings.
        # List of hypothetical questions with corresponding chunk id.
        for hyp_ques in hyperthetical_question_list:
            hypothetical_questions.append({
                "id": str(uuid.uuid4()),
                "text": hyp_ques,
                "metadata": {
                    "corresponding_chunk_id": chunk["id"]
                }
            })
    print(f"Generated {len(hypothetical_questions)} hypothetical questions")
    return hypothetical_questions

def embed_hypothetical_questions(hypothetical_questions: List[Dict]) -> List[List[float]]:
    embeddings = []
    for hypothetical_question in hypothetical_questions:
        response = client.embeddings.create(model="text-embedding-3-small", input=hypothetical_question["text"])
        embeddings.append(response.data[0].embedding)
    print(f"Generated embeddings for {len(hypothetical_questions)} hypothetical questions")
    return embeddings

def embed_chunks(chunks: List[Dict]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(model="text-embedding-3-small", input=chunk["text"])
        embeddings.append(response.data[0].embedding)
    print(f"Generated embeddings for {len(chunks)} chunks")
    return embeddings

def store_in_chromaDB(chunks: List[Dict], chunks_embeddings: List[List[float]], hyp_ques: List[Dict], hyp_ques_embedding: List[List[float]]):
    client_db = chromadb.PersistentClient(path=chroma_db_path)
    # Using cosine distance instead of euclidean distance.
    # Store chunks in ChromaDB.
    collection = client_db.get_or_create_collection(
        name="chunks_collection",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=chunks_embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks]
        )
    print("Chunks embeddings stored in ChromaDB.")
    # Store hypothetical questions in ChromaDB.
    collection = client_db.get_or_create_collection(
        name="hyp_ques_collection",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[hyp_que["id"] for hyp_que in hyp_ques],
        embeddings=hyp_ques_embedding,
        metadatas=[hyp_que["metadata"] for hyp_que in hyp_ques],
        documents=[hyp_que["text"] for hyp_que in hyp_ques]
        )
    print("Hypothetical questions embeddings stored in ChromaDB.")

def get_top_k_similar_chunks(query: str) -> List[Dict]:
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_embedding = response.data[0].embedding

    client_db = chromadb.PersistentClient(path=chroma_db_path)
    chunks_collection = client_db.get_collection(name="chunks_collection")
    hyp_ques_collection = client_db.get_collection(name="hyp_ques_collection")

    top_chunks_res = chunks_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    top_hyp_ques_res = hyp_ques_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Aggregate/de-duplicate top chunks hypothetical questions' corresponding chunks.
    aggregated_chunks: Dict[str, Dict] = {}

    for ids_group, metas_group, docs_group, dists_group in zip(
        top_chunks_res.get("ids", []),
        top_chunks_res.get("metadatas", []),
        top_chunks_res.get("documents", []),
        top_chunks_res.get("distances", []),
    ):
        for chunk_id, meta, text, dist in zip(ids_group, metas_group, docs_group, dists_group):
            cosine_sim = 1.0 - dist
            aggregated_chunks[chunk_id] = {
                "page_number": meta.get("page_number"),
                "source": meta.get("source"),
                "text": text,
                "distance": dist,
                "similarity": cosine_sim,
            }

    for metas_group, dists_group in zip(
        top_hyp_ques_res.get("metadatas", []),
        top_hyp_ques_res.get("distances", []),
    ):
        for meta, dist in zip(metas_group, dists_group):
            chunk_id = meta.get("corresponding_chunk_id")
            if not chunk_id:
                continue
            cosine_sim = 1.0 - dist
            # If the chunk_id is not in the aggregated_chunks, 
            # or the cosine_sim is greater than the aggregated_chunks[chunk_id]["similarity"], 
            # then update the aggregated_chunks.
            if chunk_id not in aggregated_chunks or cosine_sim > aggregated_chunks[chunk_id]["similarity"]:
                chunk_data = chunks_collection.get(ids=[chunk_id])
                chunk_docs = chunk_data.get("documents", [])
                chunk_metas = chunk_data.get("metadatas", [])
                if not chunk_docs:
                    continue
                chunk_meta = chunk_metas[0] if chunk_metas else {}
                aggregated_chunks[chunk_id] = {
                    "page_number": chunk_meta.get("page_number"),
                    "source": chunk_meta.get("source"),
                    "text": chunk_docs[0],
                    "distance": dist,
                    "similarity": cosine_sim,
                }

    if aggregated_chunks:
        similarities = [entry["similarity"] for entry in aggregated_chunks.values()]
        print(f"Similarity scores range: {min(similarities):.3f} to {max(similarities):.3f}")

    filtered = [
        (chunk_id, data)
        for chunk_id, data in aggregated_chunks.items()
        if data["similarity"] >= THRESHOLD
    ]
    filtered.sort(key=lambda item: item[1]["similarity"], reverse=True)

    results = []
    chunk_index = 1
    for _, data in filtered:
        results.append({
            "chunk_index": chunk_index,
            "page_number": data.get("page_number"),
            "source": data.get("source"),
            "text": data.get("text"),
            "distance": data.get("distance"),
            "similarity": data.get("similarity"),
        })
        chunk_index += 1

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
    '''
    Pipeline 1: Process PDF, create chunks, generate embeddings, and store in ChromaDB
    '''
    print("="*50)
    
    pages = file_to_pages()
    chunks = pages_to_chunks(pages)
    chunks_embeddings = embed_chunks(chunks)
    hypothetical_questions = generate_hypothetical_questions(chunks)
    hyp_ques_embeddings = embed_hypothetical_questions(hypothetical_questions)
    store_in_chromaDB(chunks, chunks_embeddings, hypothetical_questions, hyp_ques_embeddings)

    print("Embedding pipeline completed successfully!")
    print("Data is now stored in ChromaDB and ready for querying.")

    return chunks, chunks_embeddings, hypothetical_questions, hyp_ques_embeddings

def run_query_pipeline():
    """Pipeline 2: Query the ChromaDB and generate answers using LLM"""
    print("="*50)
    
    # Load data from ChromaDB
    client_db = chromadb.PersistentClient(path=chroma_db_path)
    collection = client_db.get_collection(name="chunks_collection")
    
    # Get all documents from the collection
    results = collection.get()
    chunks = results['documents']
    
    print(f"Loaded {len(chunks)} chunks from ChromaDB")
        
    # Test questions
    '''
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

    queries = ["What is NVIDIA's revenue for Q2 2025?"]



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
            chunks, chunks_embeddings, hypothetical_questions, hyp_ques_embeddings = run_embedding_pipeline()
            if chunks and chunks_embeddings and hypothetical_questions and hyp_ques_embeddings:
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
