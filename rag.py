import os
import chromadb
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()


# ==============================
# Gemini
# ==============================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


# ==============================
# ChromaDB
# ==============================

chroma_client = chromadb.PersistentClient(
    path="vector_db"
)

collection = chroma_client.get_collection(
    name="sports_documents"
)


# ==============================
# Get documents from ChromaDB
# ==============================

data = collection.get(
    include=["documents", "embeddings"]
)

documents = data["documents"]
embeddings = data["embeddings"]


# ==============================
# Create TF-IDF vectorizer
# ==============================

vectorizer = TfidfVectorizer(
    max_features=5000
)

vectorizer.fit(documents)


# ==============================
# Football RAG Function
# ==============================

def ask_football_question(question):

    # ==============================
    # Football-only check
    # ==============================

    football_keywords = [
        "ronaldo",
"cristiano ronaldo",
"messi",
"lionel messi",
"neymar",
"mbappe",
"haaland",
"pele",
"maradona",
"ronaldinho",
"zidane",
"beckham",
"premier league",
"la liga",
"bundesliga",
"serie a",
"ballon d'or"
        "football",
        "soccer",
        "goal",
        "goals",
        "offside",
        "foul",
        "penalty",
        "free kick",
        "corner",
        "red card",
        "yellow card",
        "goalkeeper",
        "defender",
        "midfielder",
        "striker",
        "referee",
        "var",
        "fifa",
        "world cup",
        "champions league",
        "premier league",
        "uefa",
        "substitution",
        "dribble",
        "tackle",
        "header",
        "kick",
        "throw in",
        "goal kick",
        "league",
        "match",
        "player",
        "coach",
        "manager",
        "formation",
        "offense",
        "defense"
        "team",
"teams",
"player",
"players",
"formation",
"position",
"positions",
"captain",
"substitute",
"substitutes",
"match",
"matches",
"stadium",
"field",
"pitch",
"ball",
"game",
"games",
"competition",
"tournament",
"league",
"club",
"clubs",
"national team",
"football team",
"team size",
"playing eleven",
"starting eleven"
    ]

    question_lower = question.lower()

    if not any(
        keyword in question_lower
        for keyword in football_keywords
    ):

        return (
            "Sorry, I can only answer "
            "football-related questions."
        )


    # ==============================
    # Convert question into vector
    # ==============================

    question_vector = vectorizer.transform(
        [question]
    ).toarray()[0]


    # ==============================
    # Calculate similarity
    # ==============================

    similarities = []

    for embedding in embeddings:

        dot_product = sum(
            a * b
            for a, b in zip(
                question_vector,
                embedding
            )
        )

        similarities.append(dot_product)


    # ==============================
    # Get top 3 relevant chunks
    # ==============================

    top_indexes = sorted(
        range(len(similarities)),
        key=lambda i: similarities[i],
        reverse=True
    )[:3]


    # ==============================
    # Combine relevant documents
    # ==============================

    relevant_text = "\n\n".join(
        documents[i]
        for i in top_indexes
    )


    # ==============================
    # Prompt for Gemini
    # ==============================

    prompt = f"""
You are FootballAI, a football-only AI assistant.

You MUST answer ONLY questions related to football.

Use ONLY the information provided in the football
knowledge base below.

--- FOOTBALL KNOWLEDGE ---
{relevant_text}
--- END FOOTBALL KNOWLEDGE ---

User Question:
{question}

Rules:

1. Answer ONLY football-related questions.

2. Do NOT answer questions about cricket,
basketball, tennis, hockey, or any other sport.

3. Do NOT answer general questions unrelated
to football.

4. If the question is not related to football,
say exactly:

"Sorry, I can only answer football-related questions."

5. Use the provided football knowledge when
answering the question.

6. Do not use your general knowledge to answer
questions outside the provided football knowledge.

7. Do not invent information.

8. Give a clear and helpful answer.

9. Keep the answer relevant to the user's question.
"""


    # ==============================
    # Generate Gemini Response
    # ==============================

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    # ==============================
    # Return Answer
    # ==============================

    return response.text


# ==============================
# Test
# ==============================

if __name__ == "__main__":

    question = input(
        "Ask a football question: "
    )

    answer = ask_football_question(
        question
    )

    print("\nFootballAI:")
    print(answer)