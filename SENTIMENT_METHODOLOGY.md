# 🎭 Sentiment Detection Methodology Explained

## How Your System Determines Positive/Negative/Neutral/Sarcastic

Your MemeCrafter system uses a **hybrid approach** combining:
1. **Fine-tuned BLIP-2 model** (learned from training data)
2. **Zero-shot Visual Question Answering (VQA)**
3. **Keyword-based parsing**

---

## 📊 The Training Data Foundation

### Original Dataset Labels (`archive (4)/labels.csv`)

Your dataset has **5 sentiment categories** with human annotations:

```
Sentiment Distribution (6,974 memes):
- positive: 3,118 (45%)
- neutral: 2,195 (31%)
- very_positive: 1,031 (15%)
- negative: 479 (7%)
- very_negative: 151 (2%)
```

**Examples from your data:**
```csv
image_1.jpg → "LOOK THERE MY FRIEND LIGHTYEAR..." → very_positive
image_6.jpg → "1998: Don't get in car with strangers..." → negative
image_7.png → "10 years challenge is lit af..." → negative
image_8.jpg → "10 Year Challenge emotional edition" → neutral
```

### What Happened During Training

When you trained BLIP-2 for 10 epochs:
- Model learned to generate **captions** from images
- It saw memes with their sentiment labels in the dataset
- Through fine-tuning, it implicitly learned:
  - What visual features correlate with different sentiments
  - What text patterns indicate humor, sarcasm, negativity
  - Context understanding (e.g., dark humor, irony)

**BUT**: The model was trained for **caption generation**, NOT direct sentiment classification!

---

## 🎯 Current Sentiment Detection Approach

### Method: Zero-Shot VQA (Visual Question Answering)

Your system asks BLIP-2 questions and parses the answers:

```python
# Step 1: Ask BLIP-2 about sentiment
sentiment_prompt = "Question: What is the sentiment of this image? Answer:"

# Step 2: BLIP-2 generates a response
sentiment_response = self.model.generate_caption(image, prompt=sentiment_prompt)
# Example response: "positive and happy"

# Step 3: Parse the response using keywords
sentiment_keywords = {
    'positive': ['positive', 'happy', 'funny', 'humorous', 'joyful', 'cheerful', 'good'],
    'negative': ['negative', 'sad', 'angry', 'bad', 'upset', 'annoyed'],
    'neutral': ['neutral', 'normal', 'calm', 'indifferent'],
    'sarcastic': ['sarcastic', 'ironic', 'satirical', 'mocking']
}

# Step 4: Count keyword matches
for sentiment, keywords in sentiment_keywords.items():
    count = sum(1 for keyword in keywords if keyword in sentiment_response)
    sentiment_scores[sentiment] = count

# Step 5: Pick sentiment with highest count
detected_sentiment = max(sentiment_scores, key=sentiment_scores.get)
```

---

## 🔍 How Each Component Works

### 1. **BLIP-2 Fine-tuned Understanding**

**What the model learned during training:**
```
Image features + OCR text + Context → Caption

The model implicitly learns:
- Irony: Image shows opposite of text
- Sarcasm: Mocking tone, exaggeration
- Positive: Happy faces, uplifting messages
- Negative: Sad faces, dark humor, critical commentary
```

**Example:**
- Image: Person crying
- Text: "Me after finishing my assignment at 3am"
- Model learns: This is humorous self-deprecation (positive despite crying)

### 2. **VQA Prompting Strategy**

**Prompt Design:**
```python
# If meme has text:
"This meme says: 'when you finally finish your project'. 
 Question: What is the sentiment? Answer:"

# If no text:
"Question: What is the sentiment of this image? Answer:"
```

**Why this works:**
- BLIP-2 was pretrained on VQA tasks
- Fine-tuning enhanced its meme understanding
- It can answer questions about images it sees

**Generation Parameters:**
```python
max_length=20       # Short, focused answer
temperature=0.3     # Low randomness
do_sample=False     # Greedy decoding (deterministic)
num_beams=3         # Beam search for quality
```

### 3. **Keyword Parsing**

**Matching Strategy:**
```python
# Count all occurrences of keywords in response
response = "positive and happy"

Matches:
- 'positive' keyword list: 2 matches (positive, happy)
- 'negative' keyword list: 0 matches
- 'neutral' keyword list: 0 matches
- 'sarcastic' keyword list: 0 matches

Result: positive (highest count)
```

**Expanded Keywords:**
```python
sentiment_keywords = {
    'positive': [
        'positive', 'happy', 'funny', 'humorous', 
        'joyful', 'cheerful', 'good'
    ],
    'negative': [
        'negative', 'sad', 'angry', 'bad', 
        'upset', 'annoyed'
    ],
    'neutral': [
        'neutral', 'normal', 'calm', 'indifferent'
    ],
    'sarcastic': [
        'sarcastic', 'ironic', 'satirical', 'mocking'
    ]
}
```

---

## ⚙️ The Decision Pipeline

```
Input Image + OCR Text
        ↓
┌───────────────────────────┐
│ 1. BLIP-2 Processes Image│
│    - Vision encoder       │
│    - Q-Former fusion      │
│    - Fine-tuned weights   │
└───────────┬───────────────┘
            ↓
┌───────────────────────────┐
│ 2. VQA Generation         │
│    "What is sentiment?"   │
│    → "positive and happy" │
└───────────┬───────────────┘
            ↓
┌───────────────────────────┐
│ 3. Keyword Matching       │
│    Count matches per      │
│    sentiment category     │
└───────────┬───────────────┘
            ↓
┌───────────────────────────┐
│ 4. Select Max Score       │
│    positive: 2 ✓          │
│    negative: 0            │
│    neutral: 0             │
│    sarcastic: 0           │
└───────────┬───────────────┘
            ↓
    Result: POSITIVE
```

---

## 📈 Confidence & Metrics

### Current Confidence Scoring

```python
'confidence': 0.85  # Fixed 85% confidence
```

**Why 85%?**
- Arbitrary baseline for VQA-based approach
- Assumes BLIP-2 is generally reliable when it gives clear answers
- NOT based on actual probability scores

### What Could Improve This

**Option 1: Use Model Logits**
```python
# Get actual probability from model's softmax output
outputs = self.model.generate(..., return_dict_in_generate=True, output_scores=True)
confidence = torch.softmax(outputs.scores[0], dim=-1).max().item()
```

**Option 2: Multiple Prompts + Voting**
```python
prompts = [
    "What is the sentiment?",
    "Is this positive or negative?",
    "What emotion does this show?"
]
# Ask all 3, see if they agree
confidence = agreement_rate
```

**Option 3: Train Sentiment Classifier Head**
```python
# Add classification layer on top of BLIP-2
class SentimentBLIP2(BLIP2):
    def __init__(self):
        super().__init__()
        self.sentiment_head = nn.Linear(hidden_dim, 4)  # 4 classes
    
# Train with your sentiment labels
loss = CrossEntropyLoss(predictions, sentiment_labels)
```

---

## 🎯 Accuracy Considerations

### Strengths ✅

1. **Multimodal Understanding**
   - Considers both image content AND text
   - Fine-tuned on 6,974 memes with sentiment labels

2. **Context-Aware**
   - Understands irony (e.g., crying but caption is funny)
   - Recognizes meme formats

3. **No Separate Classifier Needed**
   - Leverages BLIP-2's VQA capabilities
   - Unified model for all tasks

### Limitations ⚠️

1. **Indirect Classification**
   - Relies on parsing text responses
   - May miss nuances BLIP-2 understands but doesn't articulate

2. **Simplified Sentiment Space**
   - Your data has 5 categories (very_positive, positive, neutral, negative, very_negative)
   - Demo only returns 4 (positive, negative, neutral, sarcastic)
   - Lost granularity!

3. **Keyword Dependency**
   - If BLIP-2 uses synonyms not in keyword list, may fail
   - Example: "melancholic" → won't match any keyword

4. **Fixed Confidence**
   - 85% confidence is arbitrary
   - Doesn't reflect actual certainty

---

## 🔬 Alternative Approaches You Could Implement

### Approach 1: Fine-tune Classification Head ⭐ RECOMMENDED

**How:**
```python
# Add a classification layer
class MemeCrafterWithSentiment(MemeCrafterModel):
    def __init__(self):
        super().__init__()
        self.sentiment_classifier = nn.Linear(
            self.model.config.text_config.hidden_size, 
            5  # very_positive, positive, neutral, negative, very_negative
        )
    
    def classify_sentiment(self, image):
        # Get image embeddings from BLIP-2
        features = self.model.vision_model(image)
        pooled = features.last_hidden_state.mean(dim=1)
        
        # Classify
        logits = self.sentiment_classifier(pooled)
        probs = torch.softmax(logits, dim=-1)
        
        return {
            'sentiment': class_names[logits.argmax()],
            'confidence': probs.max().item(),
            'scores': probs.tolist()
        }

# Train on your labeled data
for image, label in dataloader:
    logits = model.classify_sentiment(image)
    loss = CrossEntropyLoss(logits, label)
    loss.backward()
```

**Advantages:**
- Direct sentiment classification
- Real confidence scores from softmax
- Preserves all 5 sentiment categories
- More accurate

### Approach 2: Ensemble with TextBlob/VADER

**How:**
```python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Combine BLIP-2 + text sentiment
def hybrid_sentiment(image, ocr_text):
    # BLIP-2 visual sentiment
    visual_sent = blip2_sentiment(image)
    
    # Text-based sentiment
    if ocr_text:
        vader = SentimentIntensityAnalyzer()
        text_sent = vader.polarity_scores(ocr_text)
        
        # Weighted average
        final = 0.6 * visual_sent + 0.4 * text_sent
    else:
        final = visual_sent
    
    return final
```

### Approach 3: Few-Shot Learning with Examples

**How:**
```python
# Show BLIP-2 examples in the prompt
prompt = f"""
Example 1: [happy dog image] → positive
Example 2: [crying cat] → negative
Example 3: [eye roll] → sarcastic

Now classify this meme: [your image]
Sentiment: 
"""
```

---

## 📊 Current vs. Ideal Approach

| Aspect | Current Approach | Ideal Approach |
|--------|-----------------|----------------|
| **Method** | VQA + keyword parsing | Trained classifier head |
| **Confidence** | Fixed 85% | Model softmax probabilities |
| **Categories** | 4 (losing granularity) | 5 (matching training data) |
| **Accuracy** | ~70-80% (estimated) | ~85-90% (with proper training) |
| **Interpretability** | High (text responses) | Medium (just class probabilities) |
| **Speed** | Slower (2 generations) | Faster (single forward pass) |

---

## 💡 Quick Wins to Improve Current System

### 1. Add More Keywords
```python
sentiment_keywords = {
    'positive': [
        'positive', 'happy', 'funny', 'humorous', 'joyful', 
        'cheerful', 'good', 'great', 'awesome', 'lol', 
        'hilarious', 'amusing', 'delightful', 'uplifting'
    ],
    # ... add more for each category
}
```

### 2. Use Regex for Better Matching
```python
import re
# Match whole words only
if re.search(r'\b' + keyword + r'\b', sentiment_response):
    count += 1
```

### 3. Add Confidence from Response Length
```python
# Shorter, clearer responses = higher confidence
if len(sentiment_response.split()) <= 3:
    confidence = 0.95
elif len(sentiment_response.split()) <= 5:
    confidence = 0.85
else:
    confidence = 0.70  # Longer = more uncertain
```

### 4. Sentiment Intensity
```python
# Check for intensity modifiers
if any(word in sentiment_response for word in ['very', 'extremely', 'highly']):
    intensity = 'strong'
else:
    intensity = 'moderate'
```

---

## 🎓 Summary

**Your current sentiment detection uses:**
1. **Fine-tuned BLIP-2**: Learned patterns from 6,974 labeled memes
2. **VQA Prompting**: Asks model "What is the sentiment?"
3. **Keyword Parsing**: Counts sentiment words in response
4. **Max Voting**: Picks sentiment with most keyword matches

**It works reasonably well because:**
- BLIP-2 was fine-tuned on your sentiment-labeled data
- VQA leverages its pretrained question-answering ability
- Keyword matching is simple but effective for clear responses

**But it could be better with:**
- Trained classification head (direct sentiment prediction)
- Real confidence scores from model probabilities
- All 5 sentiment categories from your data
- Better prompt engineering or ensemble methods

---

**For a research project, the current approach is good enough!** It demonstrates multimodal understanding and leverages your fine-tuning. But document these limitations in your report.

