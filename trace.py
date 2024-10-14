import json
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
import os
import argparse
import string
import re
from sentence_transformers import SentenceTransformer
import torch
import networkx as nx
import matplotlib.pyplot as plt


def build_network(results, output_file='similarity_network.png'):
    """
    Build a network graph from text similarities and visualize it with matplotlib.

    Args:
    results -- list of similarity results
    output_file -- name of the output file (default: 'similarity_network.png')

    Returns:
    A NetworkX Graph of text similarities
    """
    # Create an undirected graph (since similarity is bidirectional)
    graph = nx.Graph()

    # Add nodes and edges to the graph for each similarity result
    for result in results:
        text1 = result['dir1_file']
        text2 = result['dir2_file']
        weight = result['num_shared_shingles']

        # Add nodes if they don't exist
        if not graph.has_node(text1):
            graph.add_node(text1)
        if not graph.has_node(text2):
            graph.add_node(text2)

        # Add edge with weight (or update weight if edge already exists)
        if graph.has_edge(text1, text2):
            graph[text1][text2]['weight'] += weight
        else:
            graph.add_edge(text1, text2, weight=weight)

    # Visualize the graph
    plt.figure(figsize=(20, 20))

    # Use spring layout for node positioning
    pos = nx.spring_layout(graph, k=0.5, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=300)

    # Draw edges with width proportional to weight
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(graph, pos, width=[w / max(weights) * 5 for w in weights])

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

    # Remove axis
    plt.axis('off')

    # Save the graph
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='PNG')
    plt.close()

    return graph


def preprocess_text(text):
  """Preprocesses a given text string by removing punctuation, converting to lowercase, and removing extra whitespace.

  Args:
    text: The input text string to preprocess.

  Returns:
    The preprocessed text string.
  """
  text = re.sub(r'\s+', ' ', text).strip()
  text = text.lower()
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text


def shingle_text(text, s=5, mode="word"):
  """Converts a text into a list of shingles.

  A shingle is a contiguous subsequence of tokens (words or characters) in the text.
  This function supports both word and character shingles.

  Args:
    text: The input text string.
    s: The size of each shingle (number of words or characters). Defaults to 5.
    mode: The type of shingle to generate. Can be either 'word' or 'character'.
          Defaults to 'word'.

  Returns:
    A list of shingles extracted from the text.

  Raises:
    ValueError: If an invalid mode is provided.
  """
  if mode == "word":
    words = text.split()
    return [" ".join(words[i : i + s]) for i in range(len(words) - s + 1)]
  elif mode == "character":
    return [text[i : i + s] for i in range(len(text) - s + 1)]
  else:
    raise ValueError("Invalid mode. Choose either 'word' or 'character'.")



def process_text_files(directory, s=4, mode="word"):
    """Processes all .txt files in a given directory and its subdirectories.

    For each file, this function reads the content, preprocesses it,
    and generates shingles (word-based or character-based).
    The results are stored in a dictionary with file names as keys.

    Args:
        directory: The path to the directory containing the .txt files.
        s: The size of each shingle (number of words or characters). Defaults to 4.
        mode: The type of shingle to generate. Can be either 'word' or 'character'.
              Defaults to 'word'.

    Returns:
        A dictionary where keys are file names and values are dictionaries
        containing the "text" and "shingles" for each file.
    """
    file_data = {}
    for root, _, files in tqdm(os.walk(directory), desc=f"Processing {directory}", unit="dir"):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        text = preprocess_text(text)
                        shingles = shingle_text(text, s, mode)
                        relative_path = os.path.relpath(file_path, directory)
                        file_data[relative_path] = {
                            "text": text,
                            "shingles": shingles,
                        }
                except IOError as e:
                    print(f"Error reading {file_path}: {e}")
    return file_data


def load_signatures(signatures):
  """Loads and prepares the signatures for comparison and analysis.

  This function takes a dictionary of file signatures (as generated by
  `process_text_files`) and converts them into numpy arrays for efficient
  processing. It also generates index mappings to track the origin of each shingle.

  Args:
    signatures: A dictionary where keys are file names and values are dictionaries
                containing the "text" and "shingles" for each file.

  Returns:
    A tuple containing:
      - names: A numpy array of file names.
      - name_idx: A numpy array of indices mapping each shingle to its source file.
      - shingles: A numpy array of hashed shingle values.
      - shingle_tuples: A list of the original shingle tuples (before hashing).
  """
  names = []
  shingles = np.array([], dtype=np.int64)
  shingle_tuples = []
  shingle_lengths = []  # Store lengths of shingles per file
  for name, data in signatures.items():
    shingle_tuples.extend(data["shingles"])
    shingle = np.array([hash(d) for d in data["shingles"]], dtype=np.int64)
    shingles = np.append(shingles, shingle)
    names.append(name)
    shingle_lengths.append(len(data["shingles"]))  # Get length for each file
  name_idx = np.repeat(np.arange(len(signatures)), shingle_lengths).astype(np.int32)  # Use shingle_lengths
  return np.array(names), name_idx, shingles, shingle_tuples


def compare_signatures_fast(signatures1, signatures2, threshold):
    """
    Compares two sets of text signatures using shingling and sparse matrix multiplication.

    Args:
      signatures1: Dictionary of text signatures for the first directory.
      signatures2: Dictionary of text signatures for the second directory.
      threshold: Minimum number of shared shingles for a pair to be considered similar.

    Returns:
      A list of dictionaries, where each dictionary represents a pair of similar texts
      and includes the filenames and shared shingles.
    """
    names1, name_idx1, shingles1, _ = load_signatures(signatures1)
    names2, name_idx2, shingles2, _ = load_signatures(signatures2)
    all_shingles = np.concatenate([shingles1, shingles2], axis=0)
    unique_shingles = np.unique(all_shingles)
    hash2dense = {v: n for n, v in enumerate(unique_shingles)}
    dense1 = np.array([hash2dense[sh] for sh in shingles1], dtype=np.int32)
    dense2 = np.array([hash2dense[sh] for sh in shingles2], dtype=np.int32)

    # Create sparse matrices for efficient comparison
    matrix1 = coo_matrix((np.ones_like(dense1), (name_idx1, dense1)),
                         shape=(names1.size, unique_shingles.size))
    matrix2 = coo_matrix((np.ones_like(dense2), (dense2, name_idx2)),
                         shape=(unique_shingles.size, names2.size))

    # Compute the co-occurrence matrix (may be large for many shingles)
    cooccurence_mat = matrix1.dot(matrix2)  # Keep it sparse

    # Find indices of similar texts
    similar_texts = []
    for idx1, idx2 in tqdm(zip(*cooccurence_mat.nonzero()), desc="Comparing signatures"):
        if cooccurence_mat[idx1, idx2] >= threshold:  # Directly compare in sparse matrix
            shingle_list1 = signatures1[names1[idx1]]["shingles"]
            shingle_list2 = signatures2[names2[idx2]]["shingles"]
            common_shingles = sorted(set(shingle_list1).intersection(set(shingle_list2)))
            similar_texts.append({
                "dir1_file": names1[idx1],
                "dir2_file": names2[idx2],
                "shared_shingles": common_shingles,
                "num_shared_shingles": len(common_shingles)
            })
    return similar_texts


def compare_texts_embeddings(similar_texts, signatures1, signatures2, model, similarity_threshold):
  """Refines similarity results by comparing text embeddings.

  This function takes a list of potentially similar text pairs (identified by
  shared shingles) and further analyzes them using a SentenceTransformer model
  to calculate the cosine similarity between their embeddings.
  Pairs with a similarity score above a given threshold are retained.

  Args:
    similar_texts: A list of dictionaries, where each dictionary represents a pair
                   of potentially similar texts and contains keys like 'dir1_file',
                   'dir2_file', 'shared_shingles', and 'num_shared_shingles'.
    signatures1: A dictionary of file signatures for the first directory, as generated by
                 `process_text_files`.
    signatures2: A dictionary of file signatures for the second directory, as generated by
                 `process_text_files`.
    model: A pre-trained SentenceTransformer model used to generate text embeddings.
    similarity_threshold: The minimum cosine similarity score for a pair of texts to be
                          considered similar.

  Returns:
    A list of dictionaries, similar to the input `similar_texts`, but filtered to
    include only the pairs that pass the similarity threshold based on their embeddings.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)

  refined_results = []
  for pair in tqdm(similar_texts, desc="Comparing with SentenceTransformer"):
    text1 = signatures1[pair['dir1_file']]['text']
    text2 = signatures2[pair['dir2_file']]['text']

    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()

    if similarity > similarity_threshold:
      refined_results.append({
          'dir1_file': pair['dir1_file'],
          'dir2_file': pair['dir2_file'],
          'similarity': float(similarity),
          'shared_shingles': pair['shared_shingles'],
          'num_shared_shingles': pair['num_shared_shingles']
      })

  return refined_results


def generate_all_pairs(signatures1, signatures2):
  """Generates all possible pairs of files from two sets of signatures.

  This function takes two dictionaries of file signatures (as generated by
  `process_text_files`) and creates a list of all possible pairings between
  the files in the first set and the files in the second set.
  Each pair is represented as a dictionary with initial values for
  'shared_shingles' and 'num_shared_shingles'.

  Args:
    signatures1: A dictionary of file signatures for the first directory.
    signatures2: A dictionary of file signatures for the second directory.

  Returns:
    A list of dictionaries, where each dictionary represents a pair of files
    and contains the keys 'dir1_file', 'dir2_file', 'shared_shingles', and
    'num_shared_shingles'.
  """
  all_pairs = []
  for file1 in signatures1:
    for file2 in signatures2:
      all_pairs.append({
          "dir1_file": file1,
          "dir2_file": file2,
          "shared_shingles": [],
          "num_shared_shingles": 0
      })
  return all_pairs


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compare text files using shingling and optional SentenceTransformer.")
    parser.add_argument('--dir1', type=str, required=True, help="Path to the first directory containing .txt files")
    parser.add_argument('--dir2', type=str, required=True, help="Path to the second directory containing .txt files")
    parser.add_argument('--s', type=int, default=4, help="Number of words or characters per shingle (default: 4)")
    parser.add_argument('--mode', type=str, required=True, default="word", choices=['word', 'character'],
                        help="Shingling mode: 'word' or 'character' (default: 'word')")
    parser.add_argument('--t', type=int, default=1, help="Threshold for minimum number of shared shingles (default: 1)")
    parser.add_argument('--m', nargs='?', const="intfloat/multilingual-e5-large",
                        help="Use SentenceTransformer model. If no model is specified, use the default.")
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                        help="Similarity threshold for SentenceTransformer comparison (default: 0.85)")

    # Parse arguments
    args = parser.parse_args()

    # Always process files to get the text content and shingles
    signatures1 = process_text_files(args.dir1, s=args.s, mode=args.mode)
    signatures2 = process_text_files(args.dir2, s=args.s, mode=args.mode)

    print(f"Number of files in dir1: {len(signatures1)}")
    print(f"Number of files in dir2: {len(signatures2)}")

    use_sentence_transformer = args.m is not None
    use_shingling = args.mode is not None and args.t > 0

    # Ensure at least one comparison method is specified
    if not (use_shingling or use_sentence_transformer):
        print(f"Error: No comparison method specified. Use either shingling (--mode and --t) or SentenceTransformer (--m).")
        return

    # Perform comparison
    if use_sentence_transformer and not use_shingling:
        # If only SentenceTransformer is specified, compare all pairs
        print(f"Using SentenceTransformer without prefiltering")
        similar_texts = generate_all_pairs(signatures1, signatures2)
    else:
        # Perform shingling-based comparison
        print(f"Performing shingling-based comparison")
        similar_texts = compare_signatures_fast(signatures1, signatures2, args.t)

    print(f"Number of pairs to compare: {len(similar_texts)}")

    # If SentenceTransformer is specified, refine results
    if use_sentence_transformer:
        print(f"Using SentenceTransformer model: {args.m}")
        print(f"Similarity threshold: {args.similarity_threshold}")
        model = SentenceTransformer(args.m)
        results = compare_texts_embeddings(similar_texts, signatures1, signatures2, model, args.similarity_threshold)
    else:
        results = similar_texts

    # Save results to JSON
    with open("similarity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Display the number of results
    print(f"Number of similar text pairs found: {len(results)}")

    # Build and save the network graph
    build_network(results)

if __name__ == "__main__":
    main()
