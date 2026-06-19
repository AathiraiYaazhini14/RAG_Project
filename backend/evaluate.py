import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retrieval import answer_question
from config import GEMINI_API_KEY

# Configure Gemini for RAGAS
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY
)

gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

ragas_llm = LangchainLLMWrapper(gemini_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

# Test questions — change these to match your uploaded document
test_questions = [
    "What is machine learning?",
    "What are the types of machine learning?",
    "What are the advantages of machine learning?",
    "What is reinforcement learning?",
    "What are the disadvantages of machine learning?"
]

def run_evaluation():
    questions = []
    answers = []
    contexts = []

    print("Running RAG pipeline for each question...")
    for question in test_questions:
        print(f"  Processing: {question}")
        result = answer_question(question)
        questions.append(question)
        answers.append(result["answer"])
        contexts.append([source["excerpt"] for source in result["sources"]])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts
    })

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("\n========== RAGAS EVALUATION RESULTS ==========")
    print(f"Faithfulness:      {results['faithfulness']:.4f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.4f}")
    print(f"Context Precision: {results['context_precision']:.4f}")
    print("==============================================")
    return results

if __name__ == "__main__":
    run_evaluation()