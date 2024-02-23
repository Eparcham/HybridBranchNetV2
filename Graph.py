import spacy
import networkx as nx
# Load the spacy model
nlp = spacy.load('en_core_web_sm')  # Make sure to use the correct model for your language
# Function to extract a graph from text
def extract_graph_from_text(text):
    # Use spacy to parse the text
    doc = nlp(text)
    # Create a new graph
    graph = nx.Graph()
    # Iterate over the sentences
    for sent in doc.sents:
        # For each entity in the sentence
        for entity in sent.ents:
            # Add a node for the entity if it's not already in the graph
            if entity.text not in graph:
                graph.add_node(entity.text, label=entity.label_)
            # Iterate over all other entities in the sentence to find possible relations
            for other_entity in sent.ents:
                if entity != other_entity:
                    # Check if there's an existing edge
                    if not graph.has_edge(entity.text, other_entity.text):
                        # Add an edge between the two entities
                        graph.add_edge(entity.text, other_entity.text)

    return graph
# Example text
text = "Apple is looking at buying U.K. startup for $1 billion"
graph = extract_graph_from_text(text)
