import pandas as pd
from textblob import TextBlob 
df = pd.read_csv("data/Customer_feedback.csv")

print("No. of Feedback :" , len(df) )
print(df.head())

#Clean Customer Feedback
df['Clean_feedback'] = ( df["feedback"].
                  str.lower().str.strip()
                  )
print("\nClean_feedback:")
print(df[["feedback","Clean_feedback"]].head())

#Sentiment analysis from feedback
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["Clean_feedback"].apply(get_sentiment)
#print("\nSentiment: ")
#print(df[["feedback","sentiment"]].head(10))

#Finding Feedback Cateory
def Categorize_feedback(text):
    text = text.lower()

    if any(word in text for word in[
        "leak","leaking","packet","packaging","opened","damaged"
    ]):
        return "Packaging"
    elif any(word in text for word in[
         "stale","smell","dryness","taste","quality"
    ]):
        return "Product quality"
    elif any(word in text for word in[
        "delivery","arrived","order","arrival"
    ]):
        return "Delivery"
    elif any(word in text for word in [
        "expensive","price","cost","amount","cheap"
    ]):
        return "Pricing"
    elif any(word in text for word in[
        "support","customer service","problem"
    ]):
        return "Customer Service"
    elif any(word in text for word in[
        "love","launch","larger","flavor","flavour","good"
    ]):
        return "Growth Opportunity"
    else:
        return "Other"

df['category'] = df["Clean_feedback"].apply(Categorize_feedback)

#Assigning Prioriity to feedback
def get_priority(row):
    category = row["category"]
    sentiment = row["sentiment"]

    if category == "Packaging" and sentiment == "Negative":
       return "Critical"
    elif category == "Product quality" and sentiment == "Negative" :
        return "High"
    elif category == "Delivery" and sentiment == "Negative"  :
        return "Medium"
    elif category == "Pricing" and sentiment == "Negative":
          return "Medium"
    elif category == "Customer Service" and sentiment == "Negative"  :
                return "High"
    elif category == "Growth Opportunity":
                return "low"
    else:
         return "low"
df["Priority"] = df.apply(get_priority,axis = 1)
print("\nPriority")
print(df[["feedback","category","sentiment","Priority"]].head(20))

print("\nCategory Summary: ")
print(df['category'].value_counts())

print("\nPriority Summary: ")
print(df["Priority"].value_counts())

print("\nSentiment Summary: ")
print(df["sentiment"].value_counts())

#Finding negative issues
negative_feedback = df[df["sentiment"] == "Negative"]
print("\nNegative Issues : ")
print(negative_feedback["category"].value_counts())