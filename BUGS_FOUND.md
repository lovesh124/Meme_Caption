# 🐛 BUGS FOUND IN DEMO.PY

## Critical Bugs Causing Issues

### **BUG #1: Fine-tuned Model Never Actually Loads** ❌❌❌

**Location**: `demo.py` line 42

**Current Code**:
```python
if model_path and not use_base_model:
    print(f"Loading fine-tuned model from {model_path}")
    self.model.load_model(model_path)  # ❌ WRONG!
    self.is_finetuned = True
```

**Problem**: 
- `load_model()` is a `@classmethod` that **returns** a new model instance
- The return value is **NOT being assigned** to `self.model`
- So `self.model` keeps the base BLIP-2 weights!
- **Result**: Both `self.model` and `self.base_model` are identical base models

**Fix**:
```python
if model_path and not use_base_model:
    print(f"Loading fine-tuned model from {model_path}")
    self.model = MemeCrafterModel.load_model(model_path)  # ✅ CORRECT!
    self.is_finetuned = True
```

**Impact**: 
- This is why `visual_description` and `generated_caption` are the same!
- Both are using the same base BLIP-2 model
- Your fine-tuned weights are never being used!

---

### **BUG #2: Sentiment Prompt Causes Model to Echo** ❌

**Location**: `demo.py` line 132

**Current Code**:
```python
sentiment_prompt = "Question: What is the sentiment of this meme? Answer with one word: positive, negative, neutral, or sarcastic. Answer:"
```

**Problem**:
- BLIP-2 might echo the prompt instead of answering
- If it returns "positive, negative, neutral, or sarcastic", the parser matches 'positive' (first in dict)
- The prompt is too verbose and confusing

**Fix**:
```python
sentiment_prompt = "Question: Is this meme positive, negative, neutral, or sarcastic? Answer:"
```

Or even better:
```python
sentiment_prompt = "Question: What emotion does this meme express? Answer:"
```

---

### **BUG #3: Sentiment Parsing is Too Greedy** ⚠️

**Location**: `demo.py` lines 162-166

**Current Code**:
```python
detected_sentiment = 'neutral'  # default
for keyword, sentiment in sentiment_map.items():
    if keyword in sentiment_response:
        detected_sentiment = sentiment
        break  # Stops at FIRST match
```

**Problem**:
- If response contains multiple keywords, it always picks 'positive' (first in dict order)
- Example: "positive but also sarcastic" → detects only 'positive'

**Better Approach**:
```python
# Count exact word matches, not substring matches
words = sentiment_response.split()
sentiment_counts = {}
for word in words:
    clean_word = word.strip('.,!?').lower()
    if clean_word in sentiment_map:
        sentiment_counts[sentiment_map[clean_word]] = sentiment_counts.get(sentiment_map[clean_word], 0) + 1

if sentiment_counts:
    detected_sentiment = max(sentiment_counts, key=sentiment_counts.get)
else:
    detected_sentiment = 'neutral'
```

---

### **BUG #4: Model Needs to be in Eval Mode After Loading** ⚠️

**Location**: `demo.py` line 42 (after fix)

**Problem**:
- After loading fine-tuned model, we don't call `.eval()` on it
- The model might be in training mode

**Fix**:
```python
if model_path and not use_base_model:
    print(f"Loading fine-tuned model from {model_path}")
    self.model = MemeCrafterModel.load_model(model_path)
    self.model.to(self.device)  # ✅ Move to device
    self.model.eval()           # ✅ Set to eval mode
    self.is_finetuned = True
```

---

## Why You're Seeing These Symptoms

### Symptom 1: Visual Description = Caption (Same Output)
**Cause**: Bug #1 - Fine-tuned model never loads
- Both `self.model` and `self.base_model` are using base BLIP-2
- They generate identical captions because they're the same model!

### Symptom 2: Sentiment Shows All 4 Categories
**Cause**: Bug #2 + Bug #3 - Model echoes prompt, parser is greedy
- BLIP-2 returns: "positive, negative, neutral, or sarcastic"
- Parser matches 'positive' (first keyword found)
- But the UI might be showing something else based on the scores dict

Wait, let me check the sentiment display code...

Actually, looking at line 234-236 in demo.py:
```python
for sent_type, score in sentiment['scores'].items():
    if score > 0:
        bar = "█" * int(score * 20)
        summary_parts.append(f"   {sent_type.capitalize()}: {bar} {score:.2%}")
```

This should only show sentiments with score > 0. But if you're seeing all 4, maybe the dict iteration order is causing confusion, OR the model is returning something that triggers multiple matches.

---

## How to Fix

I'll create a fixed version of demo.py for you.

