import joblib
import re
import unicodedata
from scipy.sparse import hstack
from preprocess import clean_text


# ============================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# ============================================================

model = joblib.load(
    "models/sentiment_model_3class.pkl"
)

word_vectorizer = joblib.load(
    "models/word_tfidf_vectorizer.pkl"
)

char_vectorizer = joblib.load(
    "models/char_tfidf_vectorizer.pkl"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    # Convert Unicode characters into normal form
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Convert repeated letters
    # reallyy   -> really
    # goooood   -> good
    # amaaazing -> amazing
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# POSITIVE PHRASES
# ============================================================

POSITIVE_PHRASES = {

    "i love this": 5,
    "i love it": 5,
    "love this": 5,
    "love it": 5,

    "i really love": 5,
    "really love": 5,

    "i enjoy this": 4,
    "i really enjoy": 5,
    "really enjoy": 5,
    "enjoy this": 4,
    "enjoy it": 4,

    "very good": 4,
    "really good": 4,
    "so good": 4,
    "too good": 4,

    "very nice": 4,
    "really nice": 4,
    "so nice": 4,

    "very great": 5,
    "really great": 5,
    "so great": 5,

    "very amazing": 5,
    "really amazing": 5,
    "so amazing": 5,

    "absolutely amazing": 6,
    "absolutely fantastic": 6,
    "absolutely wonderful": 6,

    "highly recommend": 5,
    "strongly recommend": 5,
    "would recommend": 4,

    "best ever": 6,
    "one of the best": 5,

    "works perfectly": 5,
    "working perfectly": 5,
    "works great": 5,
    "working great": 5,

    "looks great": 4,
    "looks amazing": 5,
    "looks beautiful": 5,

    "feels great": 4,
    "feels good": 4,

    "made my day": 5,
    "love the experience": 5,
    "great experience": 5,

    "excellent quality": 5,
    "great quality": 4,
    "good quality": 3,

    "very happy": 5,
    "really happy": 5,
    "so happy": 5,

    "very satisfied": 5,
    "really satisfied": 5,
    "completely satisfied": 6,

    "very impressive": 5,
    "really impressive": 5,

    "good job": 4,
    "great job": 5,
    "excellent job": 5,

    "well done": 4,
    "nicely done": 4,

    "no complaints": 3,
    "nothing to complain": 3
}


# ============================================================
# NEGATIVE PHRASES
# ============================================================

NEGATIVE_PHRASES = {

    "i hate this": 6,
    "i hate it": 6,
    "hate this": 6,
    "hate it": 6,

    "really hate": 6,
    "strongly dislike": 5,

    "i dislike this": 5,
    "i dislike it": 5,

    "very bad": 5,
    "really bad": 5,
    "so bad": 5,

    "very poor": 5,
    "really poor": 5,

    "very terrible": 6,
    "really terrible": 6,

    "absolutely terrible": 7,
    "absolutely horrible": 7,
    "absolutely awful": 7,

    "worst ever": 7,
    "the worst": 6,
    "worst product": 6,
    "worst game": 6,

    "do not recommend": 6,
    "don't recommend": 6,
    "never recommend": 6,

    "would not recommend": 6,

    "does not work": 6,
    "doesn't work": 6,
    "not working": 6,
    "never works": 6,

    "works badly": 5,
    "working badly": 5,

    "poor quality": 5,
    "terrible quality": 6,
    "bad quality": 5,

    "very disappointed": 6,
    "really disappointed": 6,
    "extremely disappointed": 7,

    "very annoying": 5,
    "really annoying": 5,

    "very boring": 5,
    "really boring": 5,

    "waste of money": 7,
    "waste my money": 7,

    "not worth it": 6,
    "not worth the money": 6,

    "complete failure": 7,
    "total failure": 7,

    "made me angry": 6,
    "makes me angry": 6,

    "very unhappy": 6,
    "really unhappy": 6,

    "very frustrated": 6,
    "really frustrated": 6
}


# ============================================================
# POSITIVE WORDS
# ============================================================

POSITIVE_WORDS = {

    "love",
    "loved",
    "loving",
    "like",
    "liked",
    "enjoy",
    "enjoyed",
    "enjoying",

    "good",
    "great",
    "amazing",
    "awesome",
    "excellent",
    "fantastic",
    "wonderful",
    "brilliant",
    "perfect",

    "best",
    "better",
    "beautiful",
    "nice",
    "cool",
    "fun",
    "funny",

    "happy",
    "happier",
    "joy",
    "joyful",

    "excited",
    "exciting",

    "satisfied",
    "satisfying",

    "impressive",
    "impressed",

    "helpful",
    "useful",

    "success",
    "successful",

    "win",
    "winner",
    "winning",

    "recommend",
    "recommended",

    "pleasant",
    "positive",

    "easy",
    "smooth",
    "smoothly",

    "reliable",
    "comfortable",

    "strong",
    "fast",
    "clean",

    "smart",
    "interesting",

    "favorite",
    "favourite",

    "super",
    "wow",
    "yay",

    "thanks",
    "thankful",
    "grateful",

    "congratulations",
    "congrats",

    "peace",
    "peaceful",

    "enjoyable",
    "outstanding",
    "incredible",
    "inspiring",

    "perfectly"
}


# ============================================================
# NEGATIVE WORDS
# ============================================================

NEGATIVE_WORDS = {

    "hate",
    "hated",
    "hating",

    "bad",
    "worse",
    "worst",

    "terrible",
    "horrible",
    "awful",
    "poor",

    "dislike",
    "disliked",

    "boring",
    "bored",

    "annoying",
    "annoyed",

    "angry",
    "anger",

    "sad",
    "unhappy",

    "disappointed",
    "disappointing",
    "disappointment",

    "frustrated",
    "frustrating",
    "frustration",

    "useless",
    "worthless",

    "broken",
    "damage",
    "damaged",

    "problem",
    "problems",

    "issue",
    "issues",

    "failure",
    "failed",
    "fail",

    "wrong",
    "mistake",
    "mistakes",

    "error",
    "errors",

    "slow",
    "difficult",
    "hard",

    "expensive",

    "waste",
    "wasted",

    "scam",
    "fake",

    "ugly",

    "pain",
    "painful",

    "horrifying",
    "horrified",

    "negative",

    "complaint",
    "complaints",

    "mad",

    "poorly",
    "badly",

    "unacceptable",
    "unhelpful",

    "unsafe",
    "dangerous"
}


# ============================================================
# NEUTRAL PHRASES
# ============================================================

NEUTRAL_PHRASES = {

    "ok",
    "okay",
    "just ok",
    "just okay",

    "is ok",
    "is okay",
    "was ok",
    "was okay",

    "are ok",
    "are okay",

    "looks ok",
    "looks okay",

    "seems ok",
    "seems okay",

    "fine",
    "just fine",

    "it is fine",
    "it's fine",
    "its fine",

    "average",
    "ordinary",
    "normal",
    "usual",

    "nothing special",
    "nothing much",

    "no opinion",
    "no strong opinion",

    "not sure",
    "i don't know",
    "i do not know",

    "maybe",
    "perhaps",

    "standard",
    "same as before",
    "same as usual",

    "as usual",
    "as expected",

    "more or less",
    "so so",

    "could be better",
    "could be worse",

    "not great",
    "not terrible",

    "not bad",

    "middle",
    "moderate",

    "regular",
    "typical",

    "nothing new",
    "nothing different"
}


# ============================================================
# INTENSIFIER WORDS
# ============================================================

STRONG_INTENSIFIERS = {

    "very",
    "really",
    "extremely",
    "absolutely",
    "totally",
    "completely",
    "highly",
    "incredibly",
    "super",
    "so"
}


# ============================================================
# NEGATION WORDS
# ============================================================

NEGATION_WORDS = {

    "not",
    "never",
    "no",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "didnt",
    "didn't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "cant",
    "can't",
    "cannot",
    "wont",
    "won't"
}


# ============================================================
# EMOJI SENTIMENT
# ============================================================

POSITIVE_EMOJIS = {
    "😀",
    "😃",
    "😄",
    "😁",
    "😆",
    "😊",
    "😍",
    "🥰",
    "😘",
    "😎",
    "🤩",
    "🥳",
    "👍",
    "👏",
    "❤️",
    "❤",
    "💖",
    "💯",
    "🔥",
    "✨",
    "🎉",
    "🙂"
}


NEGATIVE_EMOJIS = {
    "😞",
    "😔",
    "😟",
    "😕",
    "🙁",
    "☹️",
    "😣",
    "😖",
    "😫",
    "😩",
    "🥺",
    "😢",
    "😭",
    "😡",
    "😠",
    "🤬",
    "👎",
    "💔",
    "😒",
    "😤",
    "🤮",
    "🤢"
}


# ============================================================
# CHECK EXACT NEUTRAL EXPRESSIONS
# ============================================================

def check_neutral(text):

    text = normalize_text(text)

    for phrase in NEUTRAL_PHRASES:

        if re.search(
            r"\b" + re.escape(phrase) + r"\b",
            text
        ):

            return True

    return False


# ============================================================
# PHRASE SCORE
# ============================================================

def phrase_score(text):

    positive_score = 0
    negative_score = 0

    for phrase, score in POSITIVE_PHRASES.items():

        if phrase in text:
            positive_score += score

    for phrase, score in NEGATIVE_PHRASES.items():

        if phrase in text:
            negative_score += score

    return positive_score, negative_score


# ============================================================
# WORD SCORE
# ============================================================

def word_score(text):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    positive_score = 0
    negative_score = 0

    for i, word in enumerate(words):

        previous_words = words[
            max(0, i - 3):i
        ]

        has_negation = any(
            w in NEGATION_WORDS
            for w in previous_words
        )

        # ------------------------------------------
        # Positive word
        # ------------------------------------------

        if word in POSITIVE_WORDS:

            if has_negation:

                negative_score += 2

            else:

                positive_score += 2


        # ------------------------------------------
        # Negative word
        # ------------------------------------------

        elif word in NEGATIVE_WORDS:

            if has_negation:

                positive_score += 2

            else:

                negative_score += 2


        # ------------------------------------------
        # Intensifier
        # ------------------------------------------

        if word in STRONG_INTENSIFIERS:

            if i + 1 < len(words):

                next_word = words[i + 1]

                if next_word in POSITIVE_WORDS:

                    positive_score += 2

                elif next_word in NEGATIVE_WORDS:

                    negative_score += 2

    return positive_score, negative_score


# ============================================================
# EMOJI SCORE
# ============================================================

def emoji_score(text):

    positive = 0
    negative = 0

    for emoji in POSITIVE_EMOJIS:

        if emoji in text:
            positive += 3

    for emoji in NEGATIVE_EMOJIS:

        if emoji in text:
            negative += 3

    return positive, negative


# ============================================================
# EXCLAMATION / EMOTION BOOST
# ============================================================

def punctuation_score(text):

    positive = 0
    negative = 0

    # Strong positive words + !
    if "!" in text:

        positive_words_found = any(
            word in text.lower()
            for word in [
                "love",
                "amazing",
                "awesome",
                "excellent",
                "great",
                "fantastic",
                "perfect",
                "wonderful",
                "best"
            ]
        )

        if positive_words_found:
            positive += 1

    return positive, negative


# ============================================================
# RULE-BASED SENTIMENT
# ============================================================

def rule_based_prediction(text):

    normalized = normalize_text(text)

    # ------------------------------------------
    # STEP 1: Neutral
    # ------------------------------------------

    if check_neutral(normalized):

        return "Neutral", 100


    # ------------------------------------------
    # STEP 2: Phrase scores
    # ------------------------------------------

    positive_phrase, negative_phrase = phrase_score(
        normalized
    )


    # ------------------------------------------
    # STEP 3: Word scores
    # ------------------------------------------

    positive_word, negative_word = word_score(
        normalized
    )


    # ------------------------------------------
    # STEP 4: Emoji scores
    # ------------------------------------------

    positive_emoji, negative_emoji = emoji_score(
        normalized
    )


    # ------------------------------------------
    # STEP 5: Punctuation
    # ------------------------------------------

    positive_punctuation, negative_punctuation = (
        punctuation_score(normalized)
    )


    # ------------------------------------------
    # TOTAL SCORE
    # ------------------------------------------

    positive_total = (
        positive_phrase
        + positive_word
        + positive_emoji
        + positive_punctuation
    )

    negative_total = (
        negative_phrase
        + negative_word
        + negative_emoji
        + negative_punctuation
    )


    # ------------------------------------------
    # Strong positive
    # ------------------------------------------

    if positive_total >= 4 and positive_total > negative_total:

        return "Positive", positive_total


    # ------------------------------------------
    # Strong negative
    # ------------------------------------------

    if negative_total >= 4 and negative_total > positive_total:

        return "Negative", negative_total


    # ------------------------------------------
    # Moderate positive
    # ------------------------------------------

    if positive_total > negative_total and positive_total >= 2:

        return "Positive", positive_total


    # ------------------------------------------
    # Moderate negative
    # ------------------------------------------

    if negative_total > positive_total and negative_total >= 2:

        return "Negative", negative_total


    # ------------------------------------------
    # No strong rule
    # ------------------------------------------

    return None, 0


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

def machine_learning_prediction(text):

    cleaned = clean_text(text)

    word_features = word_vectorizer.transform(
        [cleaned]
    )

    char_features = char_vectorizer.transform(
        [cleaned]
    )

    combined_features = hstack(
        [
            word_features,
            char_features
        ]
    )

    prediction = model.predict(
        combined_features
    )[0]

    return prediction


# ============================================================
# FINAL SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):

    # Normalize
    normalized = normalize_text(text)


    # ------------------------------------------
    # Empty input
    # ------------------------------------------

    if not normalized:

        return "Neutral"


    # ------------------------------------------
    # RULE-BASED PREDICTION
    # ------------------------------------------

    rule_result, score = rule_based_prediction(
        normalized
    )


    # ------------------------------------------
    # Strong rule result
    # ------------------------------------------

    if rule_result is not None:

        return rule_result


    # ------------------------------------------
    # MACHINE LEARNING
    # ------------------------------------------

    return machine_learning_prediction(
        normalized
    )


# ============================================================
# USER INTERFACE
# ============================================================

print()
print("================================================")
print("       SENTIMENT ANALYSIS SYSTEM")
print("================================================")

print()
print("Supported classes:")
print("1. Positive")
print("2. Negative")
print("3. Neutral")

print()
print("Model: Random Forest + Word TF-IDF + Character TF-IDF")
print("Hybrid prediction: Rules + Machine Learning")
print()


# ============================================================
# CONTINUOUS USER INPUT
# ============================================================

while True:

    user_text = input(
        "Enter a tweet (or type 'exit' to stop): "
    )

    # Exit
    if user_text.lower().strip() == "exit":

        print()
        print("Exiting Sentiment Analysis System...")
        break

    # Empty input
    if not user_text.strip():

        print("Please enter a tweet.")
        print()
        continue

    # Predict sentiment
    result = predict_sentiment(user_text)

    print()
    print("Input:", user_text)
    print("Predicted Sentiment:", result)
    print()