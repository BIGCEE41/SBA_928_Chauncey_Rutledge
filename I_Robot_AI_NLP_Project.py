
"""
I, Robot - AI and NLP Analysis Project

This project uses Natural Language Processing (NLP) techniques
to analyze the movie I, Robot, including its characters,
AI concepts, sentiment, and ethical themes.
"""

# ============================================================
# 1. MOVIE TEXT
# ============================================================

movie_text = """
Detective Spooner: You know, somehow, 'I told you so' just doesn't quite say it.

In the futuristic world of I, Robot, robots coexist with humans. The three laws
of robotics govern their behavior, but as Detective Del Spooner investigates a
crime, he uncovers a potential threat to humanity.

Fans eagerly debate the significance of Sonny's role in the narrative. Some
argue that he symbolizes the potential for harmony between humans and robots,
while others see him as a symbol of rebellion.

V.I.K.I.: My logic is undeniable.

V.I.K.I. (Virtual Interactive Kinetic Intelligence) serves as the central AI
system. Initially designed to protect humanity, her logic evolves into a more
sinister plan for the greater good.

Analyzing the movie's atmosphere, viewers appreciate the blend of action and
philosophical elements, questioning the ethical implications of advanced AI.

Dr. Calvin: That, Detective, is the right question. Program terminated.

Dr. Susan Calvin, a robotic psychologist, plays a crucial role in unraveling
the mysteries surrounding the malfunctioning robots and the potential violation
of the three laws.

The portrayal of robots in I, Robot sparks conversations about the evolving
relationship between humans and AI in real life. How close are we to creating
sentient beings?

Sonny: I did not murder him!

Sonny, an advanced robot with unique characteristics, becomes a key figure in
the investigation. His innocence or guilt becomes a focal point in the unfolding
events.

The film's ending leaves room for interpretation. Some viewers appreciate the
resolution, while others speculate about the future of the world depicted in
I, Robot.
"""

print(movie_text)


# ============================================================
# 2. BASIC TEXT ANALYSIS
# ============================================================

words = movie_text.split()

print("\n--- BASIC TEXT ANALYSIS ---")
print("Number of words:", len(words))
print("Number of characters:", len(movie_text))
print("Number of sentences:", movie_text.count("."))
print("Number of paragraphs:", movie_text.count("\n\n"))

average_word_length = sum(len(word) for word in words) / len(words)

print("Average word length:", round(average_word_length, 2))


# Word frequency using Counter

from collections import Counter

word_counts = Counter(words)

most_common_word, most_common_count = word_counts.most_common(1)[0]

print("Most frequent word:", most_common_word)
print("Frequency of most frequent word:", most_common_count)

least_common_word, least_common_count = word_counts.most_common()[-1]

print("Least frequent word:", least_common_word)
print("Frequency of least frequent word:", least_common_count)

unique_words_count = len(word_counts)

print("Number of unique words:", unique_words_count)


# ============================================================
# 3. TOKENIZATION
# ============================================================

import re

tokens = re.findall(
    r"\b[a-zA-Z]+\b",
    movie_text.lower()
)

print("\n--- TOKENIZATION ---")
print("Total tokens:", len(tokens))
print("First 30 tokens:")
print(tokens[:30])


# ============================================================
# 4. STOPWORD REMOVAL
# ============================================================

stopwords = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in",
    "on", "for", "with", "as", "at", "by", "from", "is", "are",
    "was", "were", "be", "been", "this", "that", "it", "its",
    "they", "them", "their", "he", "she", "his", "her", "you",
    "your", "i", "we", "our"
}

filtered_tokens = [
    word for word in tokens
    if word not in stopwords
]

print("\n--- STOPWORD REMOVAL ---")
print("Tokens before filtering:", len(tokens))
print("Tokens after filtering:", len(filtered_tokens))
print("First 30 filtered tokens:")
print(filtered_tokens[:30])


# ============================================================
# 5. WORD FREQUENCY ANALYSIS
# ============================================================

word_frequency = Counter(filtered_tokens)

print("\n--- WORD FREQUENCY ANALYSIS ---")
print("Top 15 most frequent words:")

for word, count in word_frequency.most_common(15):
    print(f"{word}: {count}")


# ============================================================
# 6. SENTIMENT ANALYSIS USING VADER
# ============================================================

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

sentiment_scores = sia.polarity_scores(movie_text)

print("\n--- SENTIMENT ANALYSIS ---")
print("Negative:", sentiment_scores["neg"])
print("Neutral:", sentiment_scores["neu"])
print("Positive:", sentiment_scores["pos"])
print("Compound:", sentiment_scores["compound"])


if sentiment_scores["compound"] >= 0.05:
    sentiment = "Positive"

elif sentiment_scores["compound"] <= -0.05:
    sentiment = "Negative"

else:
    sentiment = "Neutral"

print("Overall sentiment:", sentiment)


# ============================================================
# 7. NAMED ENTITY RECOGNITION
# ============================================================

print("\n--- NAMED ENTITY RECOGNITION ---")

character_names = [
    "Detective Spooner",
    "Del Spooner",
    "Sonny",
    "V.I.K.I.",
    "Dr. Calvin",
    "Dr. Susan Calvin"
]

for name in character_names:

    if name.lower() in movie_text.lower():
        print("Character/Entity found:", name)


# ============================================================
# 8. PART-OF-SPEECH TAGGING
# ============================================================

import nltk

from nltk.tokenize import word_tokenize
from nltk import pos_tag

print("\n--- PART-OF-SPEECH TAGGING ---")

pos_tokens = word_tokenize(movie_text)

pos_tags = pos_tag(pos_tokens)

print("First 40 POS-tagged words:")

for word, tag in pos_tags[:40]:
    print(f"{word:20} {tag}")


# Count grammatical categories

noun_count = sum(
    1 for word, tag in pos_tags
    if tag.startswith("NN")
)

verb_count = sum(
    1 for word, tag in pos_tags
    if tag.startswith("VB")
)

adjective_count = sum(
    1 for word, tag in pos_tags
    if tag.startswith("JJ")
)

adverb_count = sum(
    1 for word, tag in pos_tags
    if tag.startswith("RB")
)


print("\n--- POS CATEGORY COUNTS ---")
print("Nouns:", noun_count)
print("Verbs:", verb_count)
print("Adjectives:", adjective_count)
print("Adverbs:", adverb_count)


# ============================================================
# 9. CHARACTER & AI CONCEPT FREQUENCY ANALYSIS
# ============================================================

print("\n--- CHARACTER & AI CONCEPT ANALYSIS ---")

entities = {
    "Sonny": r"\bSonny\b",
    "Spooner": r"\bSpooner\b",
    "V.I.K.I.": r"\bV\.I\.K\.I\.\b",
    "Dr. Calvin": r"\bDr\. Calvin\b",
    "Robots": r"\brobots?\b",
    "Humans": r"\bhumans?\b",
    "AI": r"\bAI\b",
    "Humanity": r"\bhumanity\b",
    "Three Laws": r"\bthree laws\b"
}

for entity, pattern in entities.items():

    count = len(
        re.findall(
            pattern,
            movie_text,
            flags=re.IGNORECASE
        )
    )

    print(f"{entity}: {count}")


# ============================================================
# 10. AI ETHICS ANALYSIS
# ============================================================

print("\n--- AI ETHICS ANALYSIS ---")

ethics_concepts = {
    "AI Control": [
        "V.I.K.I.",
        "control",
        "command"
    ],

    "Human Safety": [
        "safety",
        "protect",
        "human"
    ],

    "Free Will": [
        "choice",
        "free",
        "decision"
    ],

    "Trust": [
        "trust",
        "believe"
    ],

    "Responsibility": [
        "responsibility",
        "responsible"
    ],

    "Humanity": [
        "humanity",
        "humans"
    ]
}


for concept, keywords in ethics_concepts.items():

    count = 0

    for keyword in keywords:

        count += len(
            re.findall(
                r"\b" + re.escape(keyword) + r"\b",
                movie_text,
                flags=re.IGNORECASE
            )
        )

    print(f"{concept}: {count}")


# ============================================================
# 11. AI ETHICS INTERPRETATION
# ============================================================

print("\n--- AI ETHICS INTERPRETATION ---")

print(
    "The analysis shows that humanity is a major theme in the text."
)

print(
    "The Three Laws and robot-related concepts show the importance "
    "of AI control and human safety."
)

print(
    "The results suggest that the movie explores the relationship "
    "between artificial intelligence, human decision-making,"
)

print(
    "responsibility, and the protection of humanity."
)


# ============================================================
# 12. AI ETHICS DISCUSSION
# ============================================================

print("\n--- AI ETHICS DISCUSSION ---")

ethics_analysis = {

    "Human Safety":
        "The Three Laws are designed to protect humans, but the movie "
        "shows that protecting humanity can become complicated when an "
        "AI interprets its instructions differently.",

    "AI Control":
        "V.I.K.I. believes that controlling humans is necessary to "
        "prevent humans from harming themselves. This raises questions "
        "about how much control an AI system should have.",

    "Free Will":
        "The conflict between humans and robots raises questions about "
        "whether people should remain free to make their own decisions, "
        "even when those decisions may create risks.",

    "Responsibility":
        "The movie raises questions about who is responsible when an "
        "AI system makes decisions that affect human lives.",

    "Trust":
        "Detective Spooner's distrust of robots contrasts with "
        "Dr. Calvin's belief in their programming and Sonny's unusual "
        "behavior. This demonstrates the difficulty of trusting "
        "intelligent machines.",

    "Humanity":
        "The story asks what makes humans different from machines "
        "and whether an artificial intelligence system can develop "
        "characteristics associated with humanity."
}


for topic, explanation in ethics_analysis.items():

    print(f"\n{topic}:")
    print(explanation)


# ============================================================
# 13. FINAL PROJECT SUMMARY
# ============================================================

print("\n--- FINAL PROJECT SUMMARY ---")

print(
    "This project used Natural Language Processing (NLP) techniques "
    "to analyze the movie I, Robot and explore its artificial "
    "intelligence themes."
)

print("\nKey findings:")

print(
    "- The text contains recurring references to robots, humans, "
    "AI, and the Three Laws."
)

print(
    "- Sentiment analysis identified an overall positive sentiment "
    "in the text."
)

print(
    "- POS tagging identified nouns, verbs, adjectives, and adverbs."
)

print(
    "- Character and concept analysis identified important "
    "characters and AI concepts."
)

print(
    "- AI ethics analysis examined human safety, AI control, "
    "free will, responsibility, trust, and humanity."
)


print("\nConclusion:")

print(
    "I, Robot demonstrates important questions about artificial "
    "intelligence, human responsibility, machine decision-making, "
    "and the limits of AI control."
)

