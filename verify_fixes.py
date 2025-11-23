#!/usr/bin/env python3
"""
Quick verification script to check if fixes were applied correctly
"""
import os
import re

print("="*70)
print("VERIFYING FIXES IN DEMO FILES")
print("="*70)

def check_file(filepath, checks):
    """Check if file contains expected patterns"""
    print(f"\n📄 Checking: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"   ❌ File not found!")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    all_passed = True
    for check_name, pattern, should_exist in checks:
        found = bool(re.search(pattern, content, re.MULTILINE))
        
        if should_exist and found:
            print(f"   ✅ {check_name}")
        elif not should_exist and not found:
            print(f"   ✅ {check_name}")
        else:
            print(f"   ❌ {check_name}")
            all_passed = False
    
    return all_passed

# Checks for demo.py
demo_checks = [
    ("Bug #1 Fix: Using MemeCrafterModel.load_model()", 
     r"self\.model = MemeCrafterModel\.load_model\(model_path\)", 
     True),
    
    ("Bug #1 Fix: No longer calling load_model on instance", 
     r"self\.model\.load_model\(model_path\)", 
     False),
    
    ("Bug #2 Fix: Simplified sentiment prompt",
     r"Question: What is the sentiment of this image\? Answer:",
     True),
    
    ("Bug #2 Fix: Removed verbose prompt with all sentiments",
     r"Answer with one word: positive, negative, neutral, or sarcastic",
     False),
    
    ("Bug #3 Fix: Using sentiment_keywords dict",
     r"sentiment_keywords = \{",
     True),
    
    ("Bug #3 Fix: Using better keyword matching",
     r"sentiment_scores = \{\}",
     True),
]

# Checks for simple_demo.py
simple_demo_checks = [
    ("Bug #1 Fix: Using MemeCrafterModel.load_model()",
     r"self\.model = MemeCrafterModel\.load_model\(model_path\)",
     True),
    
    ("Bug #2 Fix: Simplified sentiment prompt",
     r"Question: What is the sentiment of this image\? Answer:",
     True),
    
    ("Bug #3 Fix: Using sentiment_keywords dict",
     r"sentiment_keywords = \{",
     True),
]

# Run checks
demo_passed = check_file("demo.py", demo_checks)
simple_passed = check_file("simple_demo.py", simple_demo_checks)

print("\n" + "="*70)
print("VERIFICATION RESULTS")
print("="*70)

if demo_passed and simple_passed:
    print("✅ ALL FIXES VERIFIED SUCCESSFULLY!")
    print("\nYou can now run:")
    print("   python demo.py --model_path models/best_model --share")
    print("\nExpected improvements:")
    print("   1. Visual description ≠ meme caption (different outputs)")
    print("   2. Sentiment detection shows only ONE sentiment")
    print("   3. Fine-tuned model actually uses your trained weights")
else:
    print("❌ SOME FIXES MAY NOT HAVE APPLIED CORRECTLY")
    print("\nPlease review the files manually or re-run the fixes.")

print("="*70)

# Additional info
print("\n📊 ADDITIONAL INFO:")
print(f"   Model path exists: {os.path.exists('models/best_model')}")
if os.path.exists('models/best_model'):
    files = os.listdir('models/best_model')
    print(f"   Model files: {len(files)} files")
    if 'adapter_model.safetensors' in files:
        print("   ✅ LoRA adapter found (adapter_model.safetensors)")
    else:
        print("   ⚠️ LoRA adapter not found!")

print("\n" + "="*70)

