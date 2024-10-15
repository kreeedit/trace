## TRACE - Text Reuse Analysis and Comparison Engine


TRACE is a simple Python script that compares the similarities between different text files using two methods: shingling and sentence embeddings. It allows you to specify the directory containing the text (txt) files. It also creates a network graph of the text similarities to see the relations of the different texts. The result of the analytics is stored in a json file (_similarity_results.json_). TRACE can be useful for tasks such as plagiarism detection, document clustering, or identifying related documents in large text collections.

<img src="daffy-trace.gif" width="400" />


### USAGE

TRACE takes the following command-line arguments:

--**dir1** (required): Path to the first directory containing .txt files

--**dir2** (required): Path to the second directory containing .txt files

--**s** (optional, default=5): Number of words or characters per shingle

--**mode** (required, default="word"): Shingling mode, either 'word' or 'character'

--**t** (optional, default=1): Threshold for minimum number of shared shingles

--**m** (optional): Use SentenceTransformer model. If specified without a value, it uses the default model "intfloat/multilingual-e5-large"

--**similarity_threshold** (optional, default=0.85): Similarity threshold for SentenceTransformer comparison

These arguments allow users to customize the text comparison process, including the directories to compare, the shingling method, and whether to use additional semantic similarity analysis with a SentenceTransformer model.

```python
python trace.py --dir1 /path/to/first/directory --dir2 /path/to/second/directory --s 5 --mode word --t 3 --m "sentence-transformers/all-MiniLM-L6-v2" --similarity_threshold 0.8
```
What this command will do:

- Process all .txt files in both specified directories, creating 5-word shingles for each file.
- Perform an initial comparison using these shingles, identifying pairs of texts that share at least 3 shingles.
- For the text pairs that pass the initial shingling comparison, it will use the specified SentenceTransformer model to calculate semantic similarity between the complete docuemnts.
- Keep only the text pairs whose semantic similarity score is above 0.8.
- Generate a network graph visualizing the similarities between texts.
- Save the results to a JSON file.

##### result.json (sample)
```json
[
  {
    "dir1_file": "2Thessalonians_chapter_3.txt",
    "dir2_file": "PannHOSB_1320_IV_charter.txt",
    "similarity": 0.9021626114845276,
    "shared_shingles": [
      "domini nostri iesu christi"
    ],
    "num_shared_shingles": 1
  },
```



## Network graph
The generated GEXF file can be opened by a variety of graph visualization app, such as [Gephi](https://gephi.org/), [Cytoscape](https://cytoscape.org/), and [Orange](https://orangedatamining.com/widget-catalog/networks/networkanalysis/). These applications allow you to view the network data as a graph, and to explore the relationships between the nodes and edges. However, the script also generates a preliminary visualization picture (text_similarity_network.png)


<img src="visualization.png" width="700" />
