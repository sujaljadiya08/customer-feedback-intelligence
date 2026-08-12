from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


category_examples = {
    "Packaging": [
        "The bottle leaked",
        "The packet was damaged",
        "The package was opened",
        "The packaging is broken",
        "The product was leaking"
    ],

    "Delivery": [
        "The delivery was late",
        "My order arrived late",
        "Shipping took too long",
        "The delivery was delayed"
    ],

    "Pricing": [
        "The product is too expensive",
        "The price is too high",
        "It costs too much",
        "The product is overpriced"
    ],

    "Product Quality": [
        "The product tastes stale",
        "The product quality is poor",
        "The product smells strange",
        "The product stopped working",
        "The product caused dryness",
        "The Fragrance is too strong",
        "My skin feels dry",
        "my skin feels irritated",
        "The product doesnt work ",
        "the product has a bad smell",
        "The taste is bad",
        "The product quality is not good"
    ],

    "Customer Service": [
        "Customer support was not helpful",
        "The support team did not solve my problem",
        "Customer service was poor",
        "Customer support solved my problem quickly",
        "The support team helped me"
    ],
    "Growth Opportunity":[
        "I love this product",
        "I will buy it again",
        "Please launch a chocolate flavour",
        "please make a larger bottle"
    ]
}


# Create examples and keep their category together
all_examples = []
example_categories = []

for category, examples in category_examples.items():
    for example in examples:
        all_examples.append(example)
        example_categories.append(category)


print("Number of examples:", len(all_examples))
print("Number of categories:", len(example_categories))


# TF-IDF
vectorizer = TfidfVectorizer()
example_vectors = vectorizer.fit_transform(all_examples)


def classify_feedback(feedback):

    feedback_vector = vectorizer.transform([feedback])

    similarities = cosine_similarity(
        feedback_vector,
        example_vectors
    )[0]

    best_match_index = similarities.argmax()

    category = example_categories[best_match_index]
    score = similarities[best_match_index]

    if score < 0.40:
        category = "Needs Human Review"

    return category, score


# Test
test_feedback = [
    "Liquid came out of the bottle",
    "My order arrived very late",
    "This product costs too much",
    "The product smells bad"
]


for feedback in test_feedback:

    category, confidence = classify_feedback(feedback)

    print("\nFeedback:", feedback)
    print("Category:", category)
    print("Confidence:", round(confidence, 2))

#loading data from Customer_feedback
df = pd.read_csv("data/Customer_Feedback.csv")

results=[]

for feedback in df["feedback"]:
    category, confidence = classify_feedback(feedback)

    results.append({
        "feedback": feedback,
        "category": category,
        "confidence": round(confidence,2)
    })
results_df = pd.DataFrame(results)

print("\nActual Customer Feedback Classification :")
print(results_df)

#summary
print("\nCategory Summary: ")
print(results_df["category"].value_counts())

print("\nLow Confidence cases:")
print(results_df[results_df['confidence'] < 0.40])