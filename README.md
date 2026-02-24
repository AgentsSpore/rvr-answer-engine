# RVR Answer Engine

A multi-perspective research assistant API implementing the Retrieve-Verify-Retrieve (RVR) methodology for comprehensive question answering.

## Based on

**Paper**: RVR: Retrieve-Verify-Retrieve for Comprehensive Question Answering  
**Authors**: Deniz Qian, Hung-Ting Chen, Eunsol Choi  
**arXiv**: https://arxiv.org/abs/2602.18425v1

Traditional search systems optimize for single-best answers. RVR addresses queries that admit a wide range of valid answers by performing multi-round retrieval with verification, maximizing answer coverage.

## Problem

For complex research tasks (legal research, medical literature review, due diligence), missing even one relevant perspective can be costly. Single-round retrieval systems suffer from:
- Limited diversity in results
- Bias toward popular answers
- Poor coverage of edge cases and alternative viewpoints

## Solution

RVR performs iterative retrieval:
1. **Retrieve**: Fetch candidate documents for the query
2. **Verify**: Filter high-quality, relevant documents
3. **Retrieve**: Augment query with verified docs to find uncovered answers

This approach achieves 10%+ relative gains in complete recall compared to traditional retrieval.

## Setup

```bash
pip install -r requirements.txt
```

## Run

### Start the API server

```bash
python app.py
```

Server runs at `http://localhost:8000`

### Example API usage

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are treatments for type 2 diabetes?",
    "max_rounds": 3,
    "top_k": 5
  }'
```

### Run demo script

```bash
python demo.py
```

## API Endpoints

### POST `/search`

Perform RVR search.

**Request body**:
```json
{
  "query": "your research question",
  "max_rounds": 3,
  "top_k": 5
}
```

**Response**:
```json
{
  "query": "...",
  "rounds": [
    {
      "round": 1,
      "retrieved_count": 10,
      "verified_count": 5,
      "verified_documents": [...]
    }
  ],
  "total_verified": 12,
  "coverage_metrics": {...}
}
```

## Architecture

- **Retriever**: Sentence-BERT embedding-based retrieval
- **Verifier**: Semantic relevance scoring with threshold filtering
- **Query Augmentation**: Incorporates verified documents into subsequent queries
- **Document Corpus**: Medical research abstracts (demo data included)

## Commercial Use Cases

- **Legal Research**: Find all relevant case precedents
- **Medical Literature Review**: Comprehensive treatment option discovery
- **Due Diligence**: Uncover all risk factors and perspectives
- **Patent Search**: Identify all prior art
- **Compliance Audit**: Ensure no regulatory requirement is missed

## License

MIT