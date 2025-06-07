# -*- coding: utf-8 -*-
import arxiv
import json
import re
import dirtyjson
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, XSD
from urllib.parse import quote
import requests  # For downloading PDFs and calling Hugging Face API
import PyPDF2    # For extracting text from PDFs
import io
from tqdm import tqdm
import time
import os
from pydantic import BaseModel, ValidationError
from typing import List
from datetime import datetime

###############################################################################
# SUPPORT FUNCTIONS
###############################################################################
DEBUG = True
def debug_log(msg: str):
    if DEBUG:
        print(f"DEBUG: {msg}")

def extract_last_valid_json(response_text: str) -> dict:
    """
    Extracts the last valid JSON block from the model's response text.
    Returns {} if no valid block is found.
    """
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    blocks = pattern.findall(response_text)
    if not blocks:
        return {}
    for candidate in reversed(blocks):
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return dirtyjson.loads(candidate)
            except Exception:
                continue
    return {}

def extract_arxiv_metadata(result) -> dict:
    """
    Extracts arXiv ID, URL, DOI, title, publication date, and authors.
    """
    try:
        short_id = result.get_short_id() or ""
        entry_url = result.entry_id or ""
        doi_val = result.doi or ""
        paper_title = result.title or ""
        pub_date = ""
        if result.published:
            pub_date = str(result.published.date())
        authors_list = []
        if result.authors:
            for author in result.authors:
                authors_list.append(str(author.name))
        return {
            "paperArxivID": short_id,
            "paperArxivURL": entry_url,
            "paperDOI": doi_val,
            "paperTitle": paper_title,
            "publishedDate": pub_date,
            "authors": authors_list
        }
    except Exception as e:
        debug_log(f"Error extracting arxiv metadata: {e}")
        return {
            "paperArxivID": "", "paperArxivURL": "", "paperDOI": "",
            "paperTitle": "", "publishedDate": "", "authors": []
        }

###############################################################################
# CONFIGURATION: KEYWORDS, TAXONOMIES, ETC.
###############################################################################
ONTOLOGY_KEYWORDS = {
    "Task": [
        "classification", "regression", "detection", "segmentation", "recognition",
        "prediction", "translation", "generation", "clustering", "forecasting",
        "tracking", "summarization", "anomaly detection", "recommendation",
        "reinforcement learning"
    ],
    "Method": [
        "GPT", "CNN", "RNN", "Transformer", "BERT", "LSTM", "GAN", "SVM", "Random Forest",
        "XGBoost", "Bayesian", "KNN", "Decision Tree", "Autoencoder", "Attention",
        "RL", "ensemble", "gradient boosting", "genetic algorithm"
    ],
    "Model": [
        "GPT-3.5-turbo", "BERT-base-cased", "EfficientNet-B7", "YOLOv8n", "ViT-B/16",
        "CLIP-ViT-L/14", "PaLM-2-540B", "DistilBERT-base-uncased", "RoBERTa-large", "ChatGPT"
    ],
    "Dataset": [
        "MNIST", "CIFAR-10", "ImageNet", "COCO", "SQuAD", "PASCAL", "Cityscapes",
        "KITTI", "VOC", "OpenImages", "UCI", "IMDB", "Common Crawl"
    ],
    "Evaluation": [
        "accuracy", "f1-score", "precision", "recall", "auc", "mAP", "BLEU", "ROUGE",
        "IoU", "Dice coefficient", "RMSE", "MAE", "perplexity"
    ],
    "TrainingAlgorithm": [
        "Adam", "SGD", "RMSProp", "Adagrad", "AdamW", "LBFGS", "Adadelta", "Nadam"
    ],
    "Repository": [
        "github.com", "bitbucket.org", "gitlab.com", "huggingface.co", "sourceforge.net",
        "https://github.com/username/repo", "https://gitlab.com/username/repo", "https://huggingface.co/models/username/model"
    ],
    "ApplicationArea": [
        "ComputerVision", "NaturalLanguageProcessing", "SpeechProcessing", "TimeSeriesAnalysis",
        "Bioinformatics", "Robotics", "Healthcare", "Finance", "Cybersecurity",
        "RecommendationSystems", "SmartCities", "AutonomousVehicles", "Agriculture",
        "Education", "Remote Sensing"
    ],
    "publishedIn": [
        "NeurIPS", "ICML", "CVPR", "ECCV", "ICLR", "ACL", "EMNLP", "AAAI",
        "IJCAI", "KDD", "SIGIR", "ICCV", "MICCAI", "Journal of Machine Learning Research",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence"
    ]
}

MODEL_TYPE_TAXONOMY = [
    "MachineLearningModel",
    "RuleBasedModel",
    "TraditionalMLModel",
    "DeepLearningModel",
    "SymbolicModel",
    "HybridModel"
]

METHOD_TYPE_TAXONOMY = [
    "MachineLearningMethod",
    "RuleBasedMethod",
    "TraditionalMLMethod",
    "DeepLearningMethod",
    "SymbolicMethod",
    "HybridMethod"
]

###############################################################################
# Pydantic SCHEMA DEFINITION
###############################################################################
class PaperMetadata(BaseModel):
    paperArxivID: str = ""
    paperArxivURL: str = ""
    paperDOI: str = ""
    paperTitle: str = ""
    publishedDate: str = ""
    authors: List[str] = []
    tasks: List[str] = []
    methods: List[str] = []
    datasets: List[str] = []
    evaluations: List[str] = []
    trainingAlgorithms: List[str] = []
    repositories: List[str] = []
    publishedIn: List[str] = []
    applicationAreas: List[str] = []
    modelTitle: str = ""
    modelType: List[str] = []
    methodType: List[str] = []  # New field for method classification

###############################################################################
# SETUP HUGGING FACE API & TOKENIZER
###############################################################################
print("Setting up Hugging Face API for Mistral-7B-Instruct-v0.3...", flush=True)
try:
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # Carica solo il tokenizer localmente per la gestione dei token (leggero)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 32768

    # Ottieni il token API di Hugging Face (deve essere impostato come variabile d'ambiente)
    HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    if HF_API_TOKEN is None:
        raise Exception("Please set the HUGGINGFACE_API_KEY environment variable.")
    # URL per le chiamate alle API di inference
    HF_API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    print("Hugging Face API setup complete.\n", flush=True)
except Exception as e:
    print(f"Error setting up Hugging Face API: {e}")
    import sys
    sys.exit(1)

###############################################################################
# PDF DOWNLOAD & TEXT EXTRACTION
###############################################################################
def download_and_extract_pdf_text(pdf_url: str) -> str:
    if not pdf_url:
        return ""
    try:
        r = requests.get(pdf_url, timeout=30)
        r.raise_for_status()
        with io.BytesIO(r.content) as f:
            reader = PyPDF2.PdfReader(f)
            text_pages = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_pages.append(t)
            return "\n".join(text_pages)
    except Exception as e:
        debug_log(f"Error downloading/parsing PDF from {pdf_url}: {e}")
        return ""

def truncate_text_to_model_limit(text: str, max_tokens: int) -> str:
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = tokens[:max_tokens]
        return tokenizer.decode(truncated, skip_special_tokens=True)
    except Exception as e:
        debug_log(f"Error encoding/truncating text: {e}")
        return text

###############################################################################
# HELPER FOR PUBLISHEDIN: Create a node compliant with the T-Box
###############################################################################
def get_publication_venue_node(value: str, AI: Namespace) -> (URIRef, URIRef):
    journals = {"Journal of Machine Learning Research", "IEEE Transactions on Pattern Analysis and Machine Intelligence"}
    node = URIRef(f"http://example.org/publishedIn/{quote(value.replace(' ', '_'))}")
    if value in journals:
        return node, AI.Journal
    else:
        return node, AI.Conference

###############################################################################
# UPDATED LLM CLASSIFICATION PROMPT FOR ENTITY EXTRACTION
###############################################################################
def generate_response(prompt: str) -> dict:
    """
    Calls the Hugging Face Inference API with the given prompt and returns the extracted JSON.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 9000,
            "do_sample": False
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            debug_log(f"Error from HF API: {response.status_code} {response.text}")
            return {}
        result = response.json()
        # L'output solitamente è una lista di dict con la chiave "generated_text"
        if isinstance(result, list) and len(result) > 0:
            raw_output = result[0].get("generated_text", "")
            debug_log("Raw model output:\n" + raw_output)
            return extract_last_valid_json(raw_output)
        else:
            debug_log("Unexpected response format: " + str(result))
            return {}
    except Exception as e:
        debug_log(f"Error generating response from Hugging Face API: {e}")
        return {}

def classify_entities_with_llm(full_text: str) -> dict:
    try:
        example_tasks = ", ".join(ONTOLOGY_KEYWORDS["Task"])
        example_methods = ", ".join(ONTOLOGY_KEYWORDS["Method"])
        example_datasets = ", ".join(ONTOLOGY_KEYWORDS["Dataset"])
        example_evals = ", ".join(ONTOLOGY_KEYWORDS["Evaluation"])
        example_train_algos = ", ".join(ONTOLOGY_KEYWORDS["TrainingAlgorithm"])
        example_app_area = ", ".join(ONTOLOGY_KEYWORDS["ApplicationArea"])
        example_models = ", ".join(ONTOLOGY_KEYWORDS["Model"])
        example_repos = "https://github.com/username/repo, https://gitlab.com/username/repo, https://huggingface.co/models/username/model"
        example_publishedIn = ", ".join(ONTOLOGY_KEYWORDS["publishedIn"])
        
        prompt_entities = f"""
You are a text classification system. Given the text below, identify ONLY the explicitly mentioned fields. For each field, return only the information that is clearly and directly stated in the text.

The fields to extract are:
- tasks
- methods
- datasets
- evaluations
- trainingAlgorithms
- repositories
- publishedIn
- applicationAreas
- modelTitle

Additional Instructions:
1. For modelTitle, distinguish a specific model name from generic methods using semantic and lexical cues. Model names are usually long, complex alphanumeric strings that may include version indicators (e.g., "v1", "v2.0", "XL") and appear in contexts like "the proposed architecture called ..." or "we introduce our model named ...". In contrast, methods (e.g., "CNN", "Transformer", "GAN") are generally shorter and well-known.
2. For applicationAreas, return an application area only if the text explicitly associates it with a task. Do not infer or attach application areas by default.
3. For repositories, always prefer to return a complete URL (e.g., "https://github.com/username/repo", "https://gitlab.com/username/repo", "https://huggingface.co/models/username/model") rather than just a domain.
4. Do not include arXiv identifiers as model names.
5. Return ONLY the information explicitly mentioned in the text.
 
Example keywords (not exhaustive):
- tasks: e.g. {example_tasks}
- methods: e.g. {example_methods}
- datasets: e.g. {example_datasets}
- evaluations: e.g. {example_evals}
- trainingAlgorithms: e.g. {example_train_algos}
- applicationAreas: e.g. {example_app_area}
- repositories: e.g. {example_repos}
- publishedIn: e.g. {example_publishedIn}
- modelTitle: e.g. {example_models}

Return ONLY a valid JSON with these keys exactly:
{{
  "tasks": [],
  "methods": [],
  "datasets": [],
  "evaluations": [],
  "trainingAlgorithms": [],
  "repositories": [],
  "publishedIn": [],
  "applicationAreas": [],
  "modelTitle": ""
}}

TEXT:
{full_text}
---
Return ONLY the JSON (RFC8259-compliant).
"""
        result = generate_response(prompt_entities)
        for k in ["tasks", "methods", "datasets", "evaluations", "trainingAlgorithms", "repositories", "publishedIn", "applicationAreas", "modelTitle"]:
            if k not in result:
                result[k] = [] if k != "modelTitle" else ""
        return result
    except Exception as e:
        debug_log(f"Error in classify_entities_with_llm: {e}")
        return {"tasks": [], "methods": [], "datasets": [], "evaluations": [],
                "trainingAlgorithms": [], "repositories": [], "publishedIn": [],
                "applicationAreas": [], "modelTitle": ""}

###############################################################################
# NEW FUNCTION: CLASSIFY MODEL TYPE BASED ON THE PAPER TEXT
###############################################################################
def classify_model_type_with_llm(full_text: str) -> list:
    """
    Analyzes the text and classifies the model types strictly according to this taxonomy:
    {MODEL_TYPE_TAXONOMY}.
    
    For example, a valid JSON output might be:
    {
      "modelType": ["DeepLearningModel"]
    }
    
    Return ONLY a valid JSON with key "modelType".
    """
    try:
        model_types_joined = ", ".join(MODEL_TYPE_TAXONOMY)
        prompt_model_type = f"""
You are a text classification system. Analyze the text below and classify the model types strictly according to this taxonomy: {model_types_joined}.

Return ONLY a valid JSON with key "modelType": [].
If the text does not explicitly indicate any model type, return an empty array.

For example, a valid output might be:
{{
  "modelType": ["DeepLearningModel"]
}}

TEXT:
{full_text}
---
Return ONLY the JSON (RFC8259-compliant).
"""
        result = generate_response(prompt_model_type)
        modelType = result.get("modelType", [])
        if not isinstance(modelType, list):
            return []
        return [mt for mt in modelType if mt in MODEL_TYPE_TAXONOMY]
    except Exception as e:
        debug_log(f"Error in classify_model_type_with_llm: {e}")
        return []

###############################################################################
# NEW FUNCTION: CLASSIFY METHOD TYPE BASED ON THE PAPER TEXT
###############################################################################
def classify_method_type_with_llm(full_text: str) -> list:
    """
    Analyzes the text and classifies the methods strictly according to this taxonomy:
    {METHOD_TYPE_TAXONOMY}.
    
    For example, a valid JSON output might be:
    {
      "methodType": ["DeepLearningMethod"]
    }
    
    Return ONLY a valid JSON with key "methodType".
    """
    try:
        method_types_joined = ", ".join(METHOD_TYPE_TAXONOMY)
        prompt_method_type = f"""
You are a text classification system. Analyze the text below and classify the methods strictly according to this taxonomy: {method_types_joined}.

Return ONLY a valid JSON with key "methodType": [].
If the text does not explicitly indicate any method type, return an empty array.

For example, a valid output might be:
{{
  "methodType": ["DeepLearningMethod"]
}}

TEXT:
{full_text}
---
Return ONLY the JSON (RFC8259-compliant).
"""
        result = generate_response(prompt_method_type)
        methodType = result.get("methodType", [])
        if not isinstance(methodType, list):
            return []
        return [mt for mt in methodType if mt in METHOD_TYPE_TAXONOMY]
    except Exception as e:
        debug_log(f"Error in classify_method_type_with_llm: {e}")
        return []

###############################################################################
# RDF SERIALIZATION
###############################################################################
def safe_uri_component(text: str) -> str:
    return quote(text.replace(" ", "_"), safe="_")

def clean_local_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text)

def add_triples_to_graph(graph: Graph, paper_id: str, data: dict):
    try:
        AI = Namespace("http://example.org/ai-ontology#")
        paper_uri = URIRef(f"http://example.org/paper/{paper_id}")
        graph.add((paper_uri, RDF.type, AI.Paper))
        
        # Add data properties: ID, URL, DOI, title
        if data.get("paperArxivID", "").strip():
            graph.add((paper_uri, AI.hasTitle, Literal(data["paperArxivID"], datatype=XSD.string)))
        if data.get("paperArxivURL", "").strip():
            graph.add((paper_uri, AI.hasTitle, Literal(data["paperArxivURL"], datatype=XSD.anyURI)))
        if data.get("paperDOI", "").strip():
            graph.add((paper_uri, AI.hasTitle, Literal(data["paperDOI"], datatype=XSD.string)))
        if data.get("paperTitle", "").strip():
            graph.add((paper_uri, AI.hasTitle, Literal(data["paperTitle"], datatype=XSD.string)))
        
        # Publication date
        pub_date = data.get("publishedDate", "")
        if re.match(r"^\d{4}-\d{2}-\d{2}$", pub_date):
            graph.add((paper_uri, AI.hasDate, Literal(pub_date, datatype=XSD.date)))
        
        # Authors
        for author in data.get("authors", []):
            if author.strip():
                author_uri = URIRef(f"http://example.org/author/{safe_uri_component(author)}")
                graph.add((author_uri, RDF.type, AI.Author))
                graph.add((author_uri, AI.hasTitle, Literal(author, datatype=XSD.string)))
                graph.add((paper_uri, AI.hasAuthor, author_uri))
        
        # Simple fields: methods, datasets, evaluations, trainingAlgorithms, repositories
        for field, prop, rdf_type in [("methods", "usesMethod", AI.Method),
                                      ("datasets", "usesDataset", AI.Dataset),
                                      ("evaluations", "hasEvaluation", AI.Evaluation),
                                      ("trainingAlgorithms", "usesTrainingAlgorithm", AI.TrainingAlgorithm),
                                      ("repositories", "hasRepository", AI.Repository)]:
            for value in data.get(field, []):
                val_clean = value.strip()
                if val_clean:
                    node = URIRef(f"http://example.org/{field}/{safe_uri_component(val_clean)}")
                    graph.add((node, RDF.type, rdf_type))
                    graph.add((node, AI.hasTitle, Literal(val_clean, datatype=XSD.string)))
                    graph.add((paper_uri, AI[prop], node))
                    # For methods, add additional method type information if available.
                    if field == "methods" and data.get("methodType", []):
                        for mt in data.get("methodType", []):
                            mt_val = mt.strip()
                            if mt_val:
                                local_name = clean_local_name(mt_val)
                                if local_name:
                                    graph.add((node, RDF.type, AI[local_name]))
        
        # Handle tasks and applicationAreas.
        task_nodes = []
        for task in data.get("tasks", []):
            task_val = task.strip()
            if task_val:
                task_node = URIRef(f"http://example.org/tasks/{safe_uri_component(task_val)}")
                graph.add((task_node, RDF.type, AI.Task))
                graph.add((task_node, AI.hasTitle, Literal(task_val, datatype=XSD.string)))
                graph.add((paper_uri, AI.addressesTask, task_node))
                task_nodes.append(task_node)
        for area in data.get("applicationAreas", []):
            area_val = area.strip()
            if area_val:
                area_node = URIRef(f"http://example.org/applicationArea/{safe_uri_component(area_val)}")
                graph.add((area_node, RDF.type, AI.ApplicationArea))
                graph.add((area_node, AI.hasTitle, Literal(area_val, datatype=XSD.string)))
                for task_node in task_nodes:
                    graph.add((task_node, AI.hasApplicationArea, area_node))
        
        # Handle publishedIn: create nodes of type Conference or Journal based on the value.
        for pub in data.get("publishedIn", []):
            pub_val = pub.strip()
            if pub_val:
                pub_node, pub_type = get_publication_venue_node(pub_val, AI)
                graph.add((pub_node, RDF.type, pub_type))
                graph.add((pub_node, AI.hasTitle, Literal(pub_val, datatype=XSD.string)))
                graph.add((paper_uri, AI.publishedIn, pub_node))
        
        # Handle model: assign model title and model type nodes.
        if data.get("modelTitle", "").strip():
            model_title = data["modelTitle"].strip()
            model_uri = URIRef(f"http://example.org/model/{safe_uri_component(model_title)}")
            graph.add((model_uri, RDF.type, AI.Model))
            graph.add((model_uri, AI.hasTitle, Literal(model_title, datatype=XSD.string)))
            graph.add((paper_uri, AI.employsModel, model_uri))
            for mt in data.get("modelType", []):
                mt_val = mt.strip()
                if mt_val:
                    local_name = clean_local_name(mt_val)
                    if local_name:
                        graph.add((model_uri, RDF.type, AI[local_name]))
    except Exception as e:
        debug_log(f"Error adding triples to graph for paper {paper_id}: {e}")

###############################################################################
# CHECKPOINT: Processed Paper IDs
###############################################################################
def read_processed_ids(filename="processed_papers.txt") -> set:
    try:
        if not os.path.isfile(filename):
            return set()
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return set(line.strip() for line in lines if line.strip())
    except Exception as e:
        debug_log(f"Error reading processed_ids from {filename}: {e}")
        return set()

def append_processed_id(paper_id: str, filename="processed_papers.txt"):
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(paper_id + "\n")
    except Exception as e:
        debug_log(f"Error writing processed_id {paper_id} to {filename}: {e}")

###############################################################################
# MAIN PROGRAM
###############################################################################
def main():
    print("Starting skip-based pagination with checkpoint. All exceptions are logged, not fatal.")
    client = arxiv.Client()
    TOTAL_PAPERS = 10000
    BATCH_SIZE = 25
    output_file = "ontology_ai_llm_extracted.ttl"
    
    AI = Namespace("http://example.org/ai-ontology#")
    graph = Graph()
    graph.bind("ai", AI)
    
    processed_ids = read_processed_ids()
    page = 0
    processed_count = 0
    
    while processed_count < TOTAL_PAPERS:
        to_fetch = page + BATCH_SIZE
        print(f"\n=== PAGE: skipping first {page}, batch_size={BATCH_SIZE}, fetching {to_fetch} results ===\n")
        try:
            s = arxiv.Search(
                query="cat:cs.AI",
                max_results=to_fetch,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            try:
                all_results = list(client.results(s))
            except Exception as e:
                debug_log(f"Error retrieving results from arxiv: {e}")
                break
            
            if len(all_results) <= page:
                print("Empty page or fewer results than expected. Stopping.")
                break
            
            page_results = all_results[page : page + BATCH_SIZE]
            for result in tqdm(page_results, desc=f"Processing from skip={page} to {page + BATCH_SIZE - 1}"):
                short_id = ""
                try:
                    short_id = result.get_short_id()
                    if short_id in processed_ids:
                        debug_log(f"Skipping already processed paper: {short_id}")
                        continue
                    
                    pdf_url = result.pdf_url
                    pdf_text = download_and_extract_pdf_text(pdf_url)
                    abstract_text = result.summary
                    combined_text = abstract_text + "\n\n" + pdf_text
                    # Utilizza il tokenizer per troncare il testo al limite del modello
                    max_tokens = tokenizer.model_max_length - 1
                    truncated_text = truncate_text_to_model_limit(combined_text, max_tokens)
                    
                    arxiv_data = extract_arxiv_metadata(result)
                    entity_data = classify_entities_with_llm(truncated_text)
                    modeltype_data = classify_model_type_with_llm(truncated_text)
                    methodtype_data = classify_method_type_with_llm(truncated_text)
                    
                    final_data = {**arxiv_data, **entity_data,
                                  "modelType": modeltype_data,
                                  "methodType": methodtype_data}
                    
                    try:
                        validated = PaperMetadata(**final_data)
                        paper_dict = validated.dict()
                    except ValidationError as ve:
                        debug_log(f"Validation error on {short_id}: {ve}")
                        continue
                    
                    debug_log(f"Output JSON for {short_id}:\n{json.dumps(paper_dict, indent=2)}")
                    add_triples_to_graph(graph, short_id, paper_dict)
                    graph.serialize(destination=output_file, format="turtle")
                    debug_log(f"Serialized after {short_id} to {output_file}.")
                    
                    processed_ids.add(short_id)
                    append_processed_id(short_id)
                    processed_count += 1
                    if processed_count >= TOTAL_PAPERS:
                        break
                except Exception as ex_paper:
                    debug_log(f"Generic error processing paper {short_id}: {ex_paper}")
                    continue
        except Exception as ex_page:
            debug_log(f"Error in pagination block (page={page}): {ex_page}")
            break
        page += BATCH_SIZE
        if processed_count >= TOTAL_PAPERS:
            break
        time.sleep(3)
    
    print(f"Done. Processed {processed_count} papers in total.")

if __name__ == "__main__":
    main()
