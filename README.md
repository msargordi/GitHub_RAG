# GitHub_RAG
This repo is a RAG model to answer question from GitHub repos.

## Preview

For a quick preview, see the image below:

![Model Preview](model.png)

## Requirements

This project tested in Python 3.10. All necessary dependencies are listed in the `requirements.txt` file. You can install them using pip:
```
pip install -r requirements.txt
```

## How to Run

1. Clone this repository:
```
git clone https://github.com/msargordi/GitHub_RAG.git
cd GitHub_RAG
```

3. Run the script with Python:
```
python ollama.py [arguments]
```
or
```
python API_call.py [arguments]
```

The script accepts several optional arguments:
```
--file_types: List of file extensions to process. Default is .py .md .ipynb .sh.
--chunk_size: Chunk size for text splitting. Default is 250.
--num_docs: Number of best retrieved documents. Default is 4.
--model_name: Name of the LLM model to use. Default is "codellama:7b".
```
Example:
```
python main.py --file_types .py .md --chunk_size 300 --num_docs 5 --model_name "codellama:13b"
```

Output Example:
```
Please enter the repository URL (e.g., https://github.com/matsilv/knowledge-injection-dnn): https://github.com/matsilv/knowledge-injection-dnn                                      
Repository already exists at repos/matsilv/knowledge-injection-dnn.

Retriever is ready!
Write your question (type 'stop' to end): What are the loss fuctions used in this repo? explain them.

---Retrieve
'Finished running: retrieve:'
---Check doc relevance
---Grade: document is relevant
---Grade: document is relevant
---Grade: document is relevant
---Grade: document is relevant
'Finished running: grade_documents:'
---Generate
---Hallucination check
score:  {'score': 'yes'}
---Decision: Generation is grounded in docs
---Grade docs vs questions
---Decision: generaton addresses question
'Finished running: generate:'
_____________________________________________________________________________
('\n'
 'The loss functions used in this repository are the SBR-inspired loss and the '
 'binary cross-entropy loss. The SBR-inspired loss is a modified version of '
 'the binary cross-entropy loss that takes into account the fact that the '
 'output of the neural network is not binary, but rather a probability '
 'distribution over multiple classes. The binary cross-entropy loss is used as '
 'the primary loss function in the repository, and the SBR-inspired loss is '
 'used as an additional term to encourage the model to produce outputs that '
 'are close to the true labels.\n'
 '\n'
 'The SBR-inspired loss is computed using the following formula:\n'
 '\n'
 'L = -1/n \\* ∑ (y_true \\* log(y_pred) + (1-y_true) \\* log(1-y_pred))\n'
 '\n'
 'where y_true is the true label, y_pred is the predicted probability '
 'distribution over multiple classes, and n is the number of classes. The '
 'binary cross-entropy loss is computed using the following formula:\n'
 '\n'
 'L = -1/n \\* ∑ (y_true \\* log(y_pred) + (1-y_true) \\* log(1-y_pred))\n'
 '\n'
 'where y_true is the true label, y_pred is the predicted probability '
 'distribution over two classes, and n is the number of classes.\n'
 '\n'
 'The SBR-inspired loss is used in combination with the binary cross-entropy '
 'loss to encourage the model to produce outputs that are close to the true '
 'labels. The lambda parameter controls the weight given to the SBR-inspired '
 'loss term, and it is set to 0.1 by default.\n'
 '\n'
 'The use of these two loss functions allows the model to learn both binary '
 'classification and multi-class classification simultaneously, which can be '
 'useful in certain applications where the output space has multiple classes '
 'but the true labels are only known for a subset of those classes.')
_____________________________________________________________________________
Write your question (type 'stop' to end): 
