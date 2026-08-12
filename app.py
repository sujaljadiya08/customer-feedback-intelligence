import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI-Customer feedback Intelligence ",
    page_icon="💬",
    layout="wide"
)

st.title("AI-Powered Customer Feedback & Product Quality Intelligence")
st.subheader("Customer Feedback & Product Quality Intelligence")
st.caption("AI-Powered Customer analysis and Product Intelligence")

# -----------------------------
# Category examples
# -----------------------------
category_examples = {
    "Packaging": [
        "The bottle leaked",
        "The packet was damaged",
        "The package was opened",
        "The packaging is broken",
        "The product was leaking",
        "The package was leaking when it arrived"
    ],
    "Delivery": [
        "The delivery was late",
        "My order arrived late",
        "Shipping took too long",
        "The delivery was delayed",
        "Delivery took much longer than expected",
        "Fast delivery and nicely packed"
    ],
    "Pricing": [
        "The product is too expensive",
        "The price is too high",
        "It costs too much",
        "The product is overpriced",
        "Good product for the price"
    ],
    "Product Quality": [
        "The product tastes stale",
        "The product quality is poor",
        "The product smells strange",
        "The product stopped working",
        "The product caused dryness",
        "The fragrance is too strong",
        "My skin feels dry",
        "My skin feels irritated",
        "The product has a bad smell",
        "The taste is bad"
    ],
    "Customer Service": [
        "Customer support was not helpful",
        "The support team did not solve my problem",
        "Customer service was poor",
        "Customer support solved my problem quickly"
    ],
    "Growth Opportunity": [
        "I love this product",
        "I will buy it again",
        "Please launch a chocolate flavor",
        "Please make a larger bottle",
        "I would like a new flavor",
        "Please introduce more variants"
    ]
}

all_examples = []
example_categories = []

for category, examples in category_examples.items():
    for example in examples:
        all_examples.append(example)
        example_categories.append(category)

vectorizer = TfidfVectorizer()
example_vectors = vectorizer.fit_transform(all_examples)


def classify_feedback(feedback):
    vector = vectorizer.transform([feedback])
    similarities = cosine_similarity(vector, example_vectors)[0]

    best_index = similarities.argmax()
    category = example_categories[best_index]
    score = float(similarities[best_index])

    if score < 0.40:
        category = "Needs Human Review"

    return category, score


def get_sentiment(feedback):
    polarity = TextBlob(feedback).sentiment.polarity

    if polarity > 0.10:
        return "Positive"
    elif polarity < -0.10:
        return "Negative"
    return "Neutral"


def get_priority(category, sentiment):
    if category == "Packaging" and sentiment == "Negative":
        return "Critical"
    elif category == "Product Quality" and sentiment == "Negative":
        return "High"
    elif category == "Customer Service" and sentiment == "Negative":
        return "High"
    elif category in ["Delivery", "Pricing"] and sentiment == "Negative":
        return "Medium"
    elif category == "Growth Opportunity":
        return "Low"
    return "Low"


# -----------------------------
# Load data
# -----------------------------
try:
    df = pd.read_csv("data/customer_feedback.csv")
except FileNotFoundError:
    st.error("customer_feedback.csv not found. Make sure it is inside the data folder.")
    st.stop()

if "feedback" not in df.columns:
    st.error("The CSV must contain a 'feedback' column.")
    st.stop()

# -----------------------------
# Analyze feedback
# -----------------------------
if "category" not in df.columns:
    df["category"] = df["feedback"].apply(lambda x: classify_feedback(str(x))[0])

if "confidence" not in df.columns:
    df["confidence"] = df["feedback"].apply(lambda x: round(classify_feedback(str(x))[1], 2))

if "sentiment" not in df.columns:
    df["sentiment"] = df["feedback"].apply(get_sentiment)

if "priority" not in df.columns:
    df["priority"] = df.apply(
        lambda row: get_priority(row["category"], row["sentiment"]),
        axis=1
    )

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

if "brand" in df.columns:
    brands = ["All"] + sorted(df["brand"].dropna().unique().tolist())
    selected_brand = st.sidebar.selectbox("Brand", brands)
    if selected_brand != "All":
        filtered_df = df[df["brand"] == selected_brand].copy()
    else:
        filtered_df = df.copy()
else:
    filtered_df = df.copy()

sentiments = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

# -----------------------------
# KPI cards
# -----------------------------
st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Feedback", len(filtered_df))

negative_count = int((filtered_df["sentiment"] == "Negative").sum())
c2.metric("Negative Feedback", negative_count)

critical_count = int((filtered_df["priority"] == "Critical").sum())
c3.metric("Critical", critical_count)

review_count = int((filtered_df["category"] == "Needs Human Review").sum())
c4.metric("Human Review", review_count)

# -----------------------------
# Charts
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Feedback by Category")
    st.bar_chart(filtered_df["category"].value_counts())

with right:
    st.subheader("Feedback by Sentiment")
    st.bar_chart(filtered_df["sentiment"].value_counts())

# -----------------------------
# Priority summary
# -----------------------------
st.subheader("Priority Summary")
st.bar_chart(filtered_df["priority"].value_counts())

# -----------------------------
# Feedback table
# -----------------------------
st.subheader("Analyzed Customer Feedback")

display_columns = [
    col for col in [
        "id", "brand", "product", "channel", "feedback",
        "category", "sentiment", "priority", "confidence"
    ]
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# New feedback analyzer
# -----------------------------
st.subheader("Analyze New Feedback")

new_feedback = st.text_area(
    "Enter a customer comment",
    placeholder="Example: The bottle arrived leaking and the product was damaged."
)

if st.button("Analyze Feedback"):
    if not new_feedback.strip():
        st.warning("Please enter some feedback.")
    else:
        category, confidence = classify_feedback(new_feedback)
        sentiment = get_sentiment(new_feedback)
        priority = get_priority(category, sentiment)

        a, b, c, d = st.columns(4)
        a.metric("Category", category)
        b.metric("Sentiment", sentiment)
        c.metric("Priority", priority)
        d.metric("Confidence", f"{confidence:.2f}")

        if category == "Needs Human Review":
            st.warning("Low-confidence prediction — send this case for human review.")
        else:
            st.success("Feedback analyzed successfully.")
