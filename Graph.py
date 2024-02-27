import spacy
import networkx as nx
nlp = spacy.load('en_core_web_sm')
def extract_graph_from_text(text):
    doc = nlp(text)
    graph = nx.Graph()
    for sent in doc.sents:
        for entity in sent.ents:
            if entity.text not in graph:
                graph.add_node(entity.text, label=entity.label_)
            for other_entity in sent.ents:
                if entity != other_entity:
                    if not graph.has_edge(entity.text, other_entity.text):
                        graph.add_edge(entity.text, other_entity.text)
    return graph
text = "Apple is looking at buying U.K. startup for $1 billion"
graph = extract_graph_from_text(text)
