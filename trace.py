import os
import threading
import time
import json
import sys
import torch
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from datasketch import MinHash
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def print_progress_message(message, num_files, interval=1):
    """
    Print a progress message with an animated ellipsis.

    Arguments:
    message -- The message to print.
    num_files -- The number of total files to process.
    interval -- The interval between ellipsis animations in seconds. (default: 1)
    """
    # Create a list to keep track of the number of processed files
    processed_files = [0]

    # Define a function to run in a separate thread that updates the message
    def run():
        while getattr(thread, "do_run", True):
            for i in range(4):
                # Calculate the number of remaining files
                remaining_files = num_files - processed_files[0]
                sys.stdout.write(f"\r{message}{'.' * i} ({remaining_files} files remaining)")
                sys.stdout.flush()
                time.sleep(interval)
        # Print a newline when done to prevent overwriting the last progress message
        sys.stdout.write("\n")

    # Create a thread object and set its run method
    thread = threading.Thread(target=run)

    # Add a method to the thread object to update the number of processed files
    def update_processed_files():
        processed_files[0] += 1

    thread.update_processed_files = update_processed_files

    return thread

# Functions for both methods
def sliding_window(text, window_size, step_size):
    """
    Create a sliding window over the text

    Arguments:
    text -- input text
    window_size -- size of each window
    step_size -- step size to move for each new window

    Returns:
    List of windowed texts
    """
    return [text[i:i+window_size] for i in range(0, len(text), step_size)]

def json_pretty_print(data, filename):
    """
    Pretty print JSON data and save it to a file

    Arguments:
    data -- input data to be printed as JSON
    filename -- name of the file to save the JSON data to

    Returns:
    Pretty printed JSON string
    """
    pretty_data = json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pretty_data)

    # Return pretty printed JSON string
    return pretty_data

def build_network(similarities):
    """
    Build a network graph from text similarities + visualize it with matplotlib

    Arguments:
    similarities -- list of text similarities

    Returns:
    A NetworkX DiGraph of text similarities
    """
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes and edges to the graph for each similarity
    for similarity in similarities:
        text1 = similarity['text1_filename']
        text2 = similarity['text2_filename']

        # Add nodes if they don't exist
        if not graph.has_node(text1):
            graph.add_node(text1)
        if not graph.has_node(text2):
            graph.add_node(text2)

        # If edge already exists, increase weight, else add edge with weight 1
        if graph.has_edge(text1, text2):
            graph[text1][text2]['weight'] += 1
        else:
            graph.add_edge(text1, text2, weight=1)
            
    # Visualize the graph
    pos = nx.spring_layout(graph, k=0.5, iterations=50)
    plt.figure(figsize=(20, 20))
    
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='black', linewidths=1, font_size=10)
    
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    
    # Save the graph
    plt.savefig('text_similarity_network.png', dpi=300, format='PNG')
    
    return graph

# MinHash functions
def shingles(text, n):
    """
    Create shingles (n-grams) from text

    Arguments:
    text -- input text
    n -- n-gram size

    Returns:
    A generator that yields shingles
    """
    return (text[i:i+n] for i in range(len(text)-n+1))

def compute_minhash(set_of_shingles, num_perm=128):
    """
    Compute MinHash for a set of shingles

    Arguments:
    set_of_shingles -- a set of shingles
    num_perm -- number of permutations for MinHash

    Returns:
    MinHash of the input shingles
    """
    m = MinHash(num_perm=num_perm)
    for shingle in set_of_shingles:
        m.update(shingle.encode('utf8'))
    return m

def compute_similarity(embedding1, embedding2):
    """
    Compute the cosine similarity between two embeddings

    Arguments:
    embedding1 -- first embedding
    embedding2 -- second embedding

    Returns:
    Cosine similarity between embedding1 and embedding2
    """
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def compare_texts_minhash(directory, window_size, step_size, ngram_size, similarity_threshold=0.7):
    """
    Compare texts in a directory using MinHashsimilarity_threshold

    Arguments:
    directory -- directory containing texts
    window_size -- size of each window
    step_size -- step size to move for each new window
    ngram_size -- size of each n-gram
    num_perm -- number of permutations for MinHash
    similarity_threshold -- threshold for comparing text similarities

    Returns:
    List of dictionaries containing text comparisons
    """
    # Create a dictionary to store the MinHash of all text windows
    minhashes = {}

    # Count the number of files to process
    num_files = len([name for name in os.listdir(directory) if name.endswith(".txt")])

    # Create a progress thread and start it
    progress_thread = print_progress_message("Processing files", num_files)
    progress_thread.start()

    # Iterate over all text files in the directory
    for count, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                text = file.read()
                # Create windowed text and compute MinHash for each window
                text_windows = sliding_window(text, window_size, step_size)
                for i, window in enumerate(text_windows):
                    set_of_shingles = set(shingles(window, ngram_size))
                    minhash = compute_minhash(set_of_shingles, num_perm=128)
                    minhashes[(filename, i, window)] = minhash
            # Update the number of processed files
            progress_thread.update_processed_files()

    # Stop the progress thread
    progress_thread.do_run = False
    progress_thread.join()

    # Compare MinHashes of all pairs of windows and store if similarity is above threshold
    similarities = []
    for (filename1, i, window1), minhash1 in minhashes.items():
        for (filename2, j, window2), minhash2 in minhashes.items():
            if filename1 < filename2:  # avoid comparing a window with itself or duplicating comparisons
                minhash_similarity = minhash1.jaccard(minhash2)
                if minhash_similarity > similarity_threshold:
                    similarities.append({
                        'text1_filename': filename1,
                        'text1_window_start': i*step_size,
                        'text1_text': window1,
                        'text2_filename': filename2,
                        'text2_window_start': j*step_size,
                        'text2_text': window2,
                        'minhash_similarity': minhash_similarity,
                    })
    return similarities



# SentenceTransformer functions
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def compare_texts_embeddings(directory, window_size, step_size, model_type, similarity_threshold=0.7):
    """
    Compare texts in a directory using SentenceTransformer embeddings

    Arguments:
    directory -- directory containing texts
    window_size -- size of each window
    step_size -- step size to move for each new window
    model_type -- SentenceTransformer model type

    Returns:
    List of dictionaries containing text comparisons
    """
    # Create a dictionary to store the embeddings of all text windows
    texts = {}

    # Initialize the SentenceTransformer model
    model = SentenceTransformer(model_type)

    # Count the number of files to process
    num_files = len([name for name in os.listdir(directory) if name.endswith(".txt")])

    # Create a progress thread and start it
    progress_thread = print_progress_message("Processing files", num_files)
    progress_thread.start()

    # Iterate over all text files in the directory
    for count, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                text = file.read()
                # Create windowed text and compute embeddings for each window
                text_windows = sliding_window(text, window_size, step_size)
                for i, window in enumerate(text_windows):
                    embedding = model.encode(window)
                    texts[(filename, i)] = (window, embedding)  # store window and embedding
            # Update the number of processed files
            progress_thread.update_processed_files()

    # Stop the progress thread
    progress_thread.do_run = False
    progress_thread.join()

    # Compare embeddings of all pairs of windows and store if similarity is above threshold
    similarities = []
    for (filename1, i), (window1, embedding1) in texts.items():
        for (filename2, j), (window2, embedding2) in texts.items():
            if filename1 < filename2:  # avoid comparing a window with itself or duplicating comparisons
                similarity = compute_similarity(embedding1, embedding2)
                if similarity > similarity_threshold:
                    similarities.append({
                        'text1_filename': filename1,
                        'text1_window_start': i*step_size,
                        'text1_text': window1,
                        'text2_filename': filename2,
                        'text2_window_start': j*step_size,
                        'text2_text': window2,
                        'similarity': float(similarity),  # convert numpy float32 to python float
                    })
    return similarities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['minihash', 'sentencetransformer'], required=True)
    parser.add_argument('--model_type', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--directory', required=True)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--similarity_threshold', type=float, default=0.7)
    args = parser.parse_args()

    start_time = time.time()

    if args.model == 'minihash':
        similarities = compare_texts_minhash(args.directory, args.window_size, args.step_size, args.ngram_size, args.similarity_threshold)
    elif args.model == 'sentencetransformer':
        similarities = compare_texts_embeddings(args.directory, args.window_size, args.step_size, args.model_type, args.similarity_threshold)

    end_time = time.time()

    print("Computation Time: ", end_time - start_time)
    
    # Print results
    json_result = json_pretty_print(similarities, 'result.json')
    print(json_result)

    # Build network
    graph = build_network(similarities)
    nx.write_gexf(graph, 'text_similarity_network.gexf')


if __name__ == '__main__':
    main()