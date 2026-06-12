from retrieval import answer_question

result = answer_question("What is Aathi learning?")

print("ANSWER:", result["answer"])
print()
print("SOURCES:")
for source in result["sources"]:
    print(f"- {source['filename']} (score: {source['score']})")
    print(f"  {source['excerpt']}")