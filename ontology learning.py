import openai
import spacy
import nltk

# Download NLTK data (if not already present)
nltk.download('punkt')

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Text containing the competency questions in English
competency_text = """
Define competency questions for the AI ontology domain, analyzing them in terms of entities (classes) and relations (properties) and generating, for each, an RDF output in Turtle. Include the following CQs:

CQ1: Which papers address a specific task and which methods or models are used to solve it?
CQ2: Who are the authors who have written papers that use deep learning models, and in which conferences or journals have they been published?
CQ3: Which datasets have been used in papers related to a specific task or belonging to a specific application area (e.g., Computer Vision or Natural Language Processing)?
CQ4: Which training algorithms have been employed to train machine learning models in papers published after a certain date?
CQ5: Which papers use hybrid models that integrate symbolic and sub-symbolic approaches, and how do they combine traditional methods with deep learning?
CQ6: Which papers address specific tasks using rule-based or symbolic approaches?
CQ7: What is the temporal distribution of papers per year and how do they correlate with different application areas?
CQ8: Which conferences and journals have hosted the largest number of AI-related papers, and what relationships emerge with the type of model used?
CQ9: Which repositories host resources (e.g., datasets or code) related to specific papers or models?
CQ10: Which evaluation metrics have been used to measure model performance and how are they distributed in relation to the different methods adopted?
CQ11: What correlations exist between training techniques, the type of model employed, and the performance achieved, and how do they vary according to the application area?
CQ12: How do methodological choices evolve over time (e.g., transition from traditional methods to deep learning or hybrid models) and what impact do they have on performance and evaluations?
"""

# Use spaCy to analyze the text and print recognized entities
doc = nlp(competency_text)
print("Recognized entities (spaCy):")
for ent in doc.ents:
    print(f" - {ent.text} ({ent.label_})")

# Use NLTK to tokenize the text into words
tokens = nltk.word_tokenize(competency_text)
print("\nFirst 20 tokens (NLTK):")
print(tokens[:20])

# Build an enriched prompt with a brief summary of the analysis
analysis_summary = (
    "Preliminary analysis (entity extraction and tokenization): "
    f"Identified entities: {[ent.text for ent in doc.ents]}. "
    f"Total number of tokens: {len(tokens)}."
)

refined_prompt = (
    competency_text +
    "\n\n" +
    "Using the semantic analysis obtained, identify the key entities and relations, and produce the RDF triples in Turtle format for each competency question.\n\n" +
    analysis_summary
)

def query_gpt_for_rdf(prompt_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in ontologies, RDF, NLP, and Python programming."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.0,
        max_tokens=3000
    )
    return response.choices[0].message["content"]

if __name__ == '__main__':
    # Set your OpenAI API key
    openai.api_key = "YOUR_OPENAI_API_KEY"
    
    # Execute the query and print the output from GPT
    output = query_gpt_for_rdf(refined_prompt)
    print("\nOutput from GPT:")
    print(output)
