from fastapi import FastAPI, File, UploadFile , Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import shutil
from io import BytesIO
import fitz 
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Create database table if it doesn't exist
def init_db():
    try:
        logger.info("Attempting to connect to database...")
        logger.info(f"Database host: {os.environ['DB_HOST']}")
        logger.info(f"Database name: {os.environ['DB_NAME']}")
        logger.info(f"Database user: {os.environ['DB_USER']}")
        logger.info(f"Database port: {os.environ['DB_PORT']}")
        
        conn = psycopg2.connect(
            dbname=os.environ['DB_NAME'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PSWD'],
            host=os.environ['DB_HOST'],
            port=os.environ['DB_PORT']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Test the connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"Connected to PostgreSQL database. Version: {version[0]}")
        
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS files (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            file_content TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        logger.info("Database table created successfully")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {str(e)}")
        logger.error("Please check your database credentials and connection settings")
        raise
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

# Initialize database
try:
    conn = init_db()
    logger.info("Database connection successful")
except Exception as e:
    logger.error(f"Database connection failed: {str(e)}")
    conn = None

app = FastAPI()
app.conversation = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application")
    # Verify Gemini API key
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not found in environment variables")
    else:
        logger.info("Gemini API key found")
        # Test API URL with a simple request
        try:
            test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
            response = requests.get(test_url)
            if response.status_code == 200:
                logger.info("Successfully connected to Gemini API")
            else:
                logger.error(f"Gemini API test failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
        except Exception as e:
            logger.error(f"Error testing Gemini API connection: {str(e)}")

def get_pdf_text(contents):
    # Process the PDF file using PyMuPDF
    pdf_document = fitz.open(stream=BytesIO(contents), filetype="pdf")
    # Initialize an empty string to store the extracted text
    extracted_text = ""
    
    # Iterate over each page and extract text
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        extracted_text += page.get_text()
    extracted_text = '\n'.join(line for line in extracted_text.splitlines() if line.strip())
    return extracted_text
   

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        logger.info("Initializing HuggingFace embeddings...")
        # Using a simpler model that's more likely to work out of the box
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        logger.info("Creating FAISS vector store...")
        vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        logger.info("Vector store created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Error in get_vectorstore: {str(e)}")
        raise

class ConversationChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.conversation_history = []
        self.api_url = GEMINI_API_URL
        
        # Test the API connection
        try:
            self.test_gemini_api()
            self.use_gemini = True
            logger.info("Successfully connected to Gemini API")
        except Exception as e:
            logger.warning(f"Failed to connect to Gemini API: {str(e)}")
            logger.warning("Using fallback response mode instead")
            self.use_gemini = False
    
    def test_gemini_api(self):
        """Test the Gemini API connection"""
        test_data = {
            "contents": [{
                "parts": [{"text": "Hello, testing API connection"}]
            }]
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, json=test_data)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        return True
        
    def invoke(self, question_dict):
        try:
            question = question_dict["question"]
            
            # Get relevant context from vectorstore
            docs = self.vectorstore.similarity_search(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            if self.use_gemini:
                try:
                    # Prepare prompt with context
                    prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease provide a helpful answer based on the context above. If the context doesn't contain relevant information, say so."
                    
                    # Prepare request data
                    data = {
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }]
                    }
                    
                    # Send request to Gemini API
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(self.api_url, headers=headers, json=data)
                    
                    # Process response
                    if response.status_code == 200:
                        response_json = response.json()
                        if "candidates" in response_json and len(response_json["candidates"]) > 0:
                            # Extract text from first candidate
                            candidate = response_json["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                answer = "".join([part.get("text", "") for part in parts])
                            else:
                                answer = "Received empty response from Gemini API"
                        else:
                            answer = "No valid response from Gemini API"
                    else:
                        logger.error(f"API request failed: {response.status_code} - {response.text}")
                        answer = f"API request failed with status code {response.status_code}. Please check your API key."
                        
                except Exception as e:
                    logger.error(f"Gemini API error: {str(e)}")
                    answer = f"Sorry, I couldn't process your question due to an API error: {str(e)}. Here's some context that might help: {context[:500]}..."
            else:
                # Fallback mode - simple response with context
                answer = f"Your question was: {question}\n\nHere's some relevant context from your document: {context[:500]}..."
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return {
                "chat_history": self.conversation_history,
                "answer": answer
            }
        except Exception as e:
            logger.error(f"Error in conversation chain: {str(e)}")
            # Return a helpful error message
            error_msg = f"Error processing your question: {str(e)}"
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return {
                "chat_history": self.conversation_history,
                "answer": error_msg
            }

def get_conversation_chain(vectorstore):
    try:
        logger.info("Initializing conversation chain with Gemini...")
        return ConversationChain(vectorstore)
    except Exception as e:
        logger.error(f"Error setting up conversation chain: {str(e)}")
        raise

def store_raw_text(raw_text, filename):
    try:
        logger.info("Attempting to store text in database...")
        cursor = conn.cursor()
        insert_query = "INSERT INTO files (filename, file_content) VALUES (%s,%s);"
        cursor.execute(insert_query, (filename, raw_text))
        conn.commit()
        logger.info("Successfully stored text in database")
    except psycopg2.Error as e:
        logger.error(f"Database error while storing text: {str(e)}")
        logger.error(f"Error code: {e.pgcode}")
        logger.error(f"Error message: {e.pgerror}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while storing text: {str(e)}")
        raise

@app.post("/process_text")
async def process_text(request: Request):
    raw_body = await request.body()
    text = raw_body.decode()
    json_data = json.loads(text)  
    print(json_data.get("question"))
    processed_text = text.upper()  # Example processing, converting text to uppercase
    return {"processed_text": processed_text}


@app.post("/chat")
async def chat(request: Request):
    try:
        raw_body = await request.body()
        text = raw_body.decode()
        json_data = json.loads(text)  
        question = json_data.get("question")
        
        if not app.conversation:
            logger.error("Conversation chain not initialized")
            return JSONResponse(
                content={"message": "Please upload a PDF first"},
                status_code=400
            )
            
        try:
            response = app.conversation.invoke({"question": question})
            return response["chat_history"]
        except Exception as e:
            logger.error(f"Error during conversation chain invoke: {str(e)}")
            # Return a fallback response instead of an error
            fallback_response = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Sorry, I encountered an error processing your question: {str(e)}. Please try again or upload a different PDF."}
            ]
            return fallback_response
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return JSONResponse(
            content={"message": f"Error processing chat request: {str(e)}"},
            status_code=500
        )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload request for: {file.filename}")
        
        # Verify database connection
        if conn is None:
            logger.error("Database connection is not available")
            return JSONResponse(
                content={"message": "Database connection is not available"},
                status_code=500
            )
        
        try:
            # Test database connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            logger.info("Database connection test successful")
        except psycopg2.Error as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return JSONResponse(
                content={"message": "Database connection test failed"},
                status_code=500
            )
        
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            return JSONResponse(
                content={"message": "Only PDF files are allowed"},
                status_code=400
            )

        contents = await file.read()
        if not contents:
            logger.warning("Empty file received")
            return JSONResponse(
                content={"message": "Empty file received"},
                status_code=400
            )

        try:
            app.conversation = None
            load_dotenv()
            
            # Verify Gemini API key
            if not GEMINI_API_KEY:
                logger.error("Gemini API key not found")
                raise ValueError("Gemini API key not found in environment variables")
            
            # get raw text
            logger.info("Extracting text from PDF...")
            try:
                raw_text = get_pdf_text(contents)
                logger.info(f"Successfully extracted {len(raw_text)} characters from PDF")
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {str(e)}")
                raise

            if not raw_text:
                logger.warning("No text extracted from PDF")
                return JSONResponse(
                    content={"message": "Could not extract text from PDF"},
                    status_code=400
                )

            try:
                #store raw text in db for record
                if conn is None:
                    logger.warning("Database connection not available, skipping storage")
                else:
                    logger.info("Storing text in database...")
                    store_raw_text(raw_text, file.filename)
                    logger.info("Successfully stored text in database")
            except psycopg2.Error as db_error:
                logger.error(f"Database error: {str(db_error)}")
                # Continue even if database storage fails
                pass

            #get text chunks
            logger.info("Creating text chunks...")
            try:
                text_chunks = get_text_chunks(raw_text)
                logger.info(f"Created {len(text_chunks)} text chunks")
            except Exception as e:
                logger.error(f"Error creating text chunks: {str(e)}")
                raise

            if not text_chunks:
                logger.warning("No text chunks created")
                return JSONResponse(
                    content={"message": "Could not create text chunks"},
                    status_code=500
                )

            # get vectorstore
            logger.info("Creating vector store...")
            try:
                vectorstore = get_vectorstore(text_chunks)
                logger.info("Successfully created vector store")
            except Exception as e:
                logger.error(f"Error creating vector store: {str(e)}")
                raise
            
            logger.info("Setting up conversation chain...")
            try:
                app.conversation = get_conversation_chain(vectorstore)
                logger.info("Successfully set up conversation chain")
            except Exception as e:
                logger.error(f"Error setting up conversation chain: {str(e)}")
                raise

            logger.info("PDF processing completed successfully")
            return JSONResponse(content={"message": "Success"}, status_code=200)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return JSONResponse(
                content={"message": f"Error processing PDF: {str(e)}"},
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            content={"message": f"Upload failed: {str(e)}"},
            status_code=500
        )
