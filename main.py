import re 
import os
import ast
import time
import random
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
llm = Ollama(model="mistral", request_timeout=3600.0)
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="Provides documentation about code for an API. Use this for reading docs for the API.",
        ),
    ),
    code_reader,
]
code_llm = Ollama(model="codellama", request_timeout=3600.0)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

def exponential_backoff(retries):
    """Implements exponential backoff with jitter and logging."""
    base_delay = 1  # 1 second base delay
    max_delay = 60  # maximum delay of 60 seconds
    delay = min(max_delay, base_delay * (2 ** retries))  # exponential increase
    jitter = random.uniform(0, 1)  # add jitter to avoid collision
    time.sleep(delay + jitter)
    print(f"Retrying in {delay + jitter:.2f} seconds...")

def is_valid_filename(filename):
    """Validate the filename to prevent directory traversal and invalid characters."""
    # Define allowed characters and a reasonable length limit
    allowed_characters = re.compile(r'^[\w\-. ]+$')  # Letters, digits, underscores, hyphens, periods, spaces
    max_length = 255  # Maximum filename length

    if len(filename) > max_length:
        print("Filename is too long.")
        return False
    if not allowed_characters.match(filename):
        print("Filename contains invalid characters.")
        return False
    return True

def process_prompt(prompt):
    """Process a single prompt with retry logic."""
    retries = 0
    max_retries = 5  # maximum number of retries

    while retries < max_retries:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            
            if all(key in cleaned_json for key in ["code", "description", "filename"]):
                return cleaned_json
            
            raise KeyError("One of the required keys ('code', 'description', 'filename') is missing in the response.")

        except KeyError as e:
            print(f"KeyError: {e}")
            return None
        except Exception as e:
            retries += 1
            print(f"Error occurred, retry #{retries}: {e}")
            if retries < max_retries:
                exponential_backoff(retries)
            else:
                print("Max retries reached, unable to process the request.")
                return None

def save_code(cleaned_json):
    """Save the generated code to a file."""
    filename = cleaned_json["filename"]
    if not is_valid_filename(filename):
        print("Invalid filename, skipping file save.")
        return
    
    output_path = os.path.join("output", filename)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except Exception as e:
        print(f"Error saving file: {e}")

# Main loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    cleaned_json = process_prompt(prompt)
    
    if cleaned_json:
        print("Code generated:")
        print(cleaned_json["code"])
        print("\n\nDescription:")
        print(cleaned_json["description"])
        save_code(cleaned_json)
    else:
        print("Unable to process request, try again later...")