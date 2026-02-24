"""Demo script showing RVR in action"""

import json
from rvr_engine import RVREngine
from document_store import DocumentStore


def print_section(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def main():
    print_section("RVR Answer Engine Demo")
    print("Based on: https://arxiv.org/abs/2602.18425v1")
    
    # Initialize
    print("\nInitializing document store and RVR engine...")
    doc_store = DocumentStore()
    doc_store.load_sample_documents()
    rvr_engine = RVREngine(doc_store)
    
    print(f"Loaded {len(doc_store.documents)} medical research documents")
    
    # Example query
    query = "What are treatments for type 2 diabetes?"
    print_section("Query")
    print(f"Q: {query}")
    
    # Perform RVR search
    print_section("RVR Search - Multi-Round Retrieval")
    result = rvr_engine.search(
        query=query,
        max_rounds=3,
        top_k=4
    )
    
    # Display results by round
    for round_info in result['rounds']:
        round_num = round_info['round']
        print(f"\n--- Round {round_num} ---")
        print(f"Retrieved: {round_info['retrieved_count']} candidates")
        print(f"Verified: {round_info['verified_count']} documents\n")
        
        for i, doc in enumerate(round_info['verified_documents'], 1):
            print(f"{i}. {doc['document']['title']}")
            print(f"   Relevance: {doc['score']:.3f}")
            print(f"   {doc['document']['text'][:150]}...\n")
    
    # Summary
    print_section("Summary")
    print(f"Total verified documents: {result['total_verified']}")
    print(f"Coverage metrics:")
    for metric, value in result['coverage_metrics'].items():
        print(f"  - {metric}: {value:.3f}")
    
    # Compare with single-round retrieval
    print_section("Comparison: Single-Round vs RVR")
    single_round = rvr_engine.search(query=query, max_rounds=1, top_k=4)
    print(f"Single-round retrieval: {single_round['total_verified']} documents")
    print(f"RVR (3 rounds): {result['total_verified']} documents")
    
    improvement = ((result['total_verified'] - single_round['total_verified']) / 
                   single_round['total_verified'] * 100)
    print(f"Improvement: +{improvement:.1f}% more comprehensive coverage")
    
    print_section("Demo Complete")
    print("\nRVR demonstrates superior answer coverage through iterative")
    print("retrieval and verification, crucial for comprehensive research.")
    print("\nTo start the API: python app.py")


if __name__ == "__main__":
    main()
