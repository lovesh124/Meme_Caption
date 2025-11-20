"""
Evaluation script for MemeCrafter
Includes multiple metrics: BLEU, ROUGE, CLIPScore, Perplexity
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
from transformers import CLIPModel, CLIPProcessor

from src.config import config
from src.model import MemeCrafterModel
from src.dataset import MemeDataset


class MemeEvaluator:
    """Evaluator for MemeCrafter model"""
    
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load CLIP for CLIPScore
        print("Loading CLIP for CLIPScore evaluation...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.results = defaultdict(list)
        
    def calculate_bleu(self, references, hypotheses):
        """Calculate BLEU scores"""
        # Tokenize
        refs_tokens = [[ref.split()] for ref in references]
        hyps_tokens = [hyp.split() for hyp in hypotheses]
        
        # Corpus BLEU
        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1/n] * n + [0] * (4-n))
            bleu_scores[f'bleu_{n}'] = corpus_bleu(refs_tokens, hyps_tokens, weights=weights)
        
        return bleu_scores
    
    def calculate_rouge(self, references, hypotheses):
        """Calculate ROUGE scores"""
        rouge_scores = defaultdict(list)
        
        for ref, hyp in zip(references, hypotheses):
            scores = self.rouge_scorer.score(ref, hyp)
            for key, value in scores.items():
                rouge_scores[f'{key}_f'].append(value.fmeasure)
                rouge_scores[f'{key}_p'].append(value.precision)
                rouge_scores[f'{key}_r'].append(value.recall)
        
        # Average scores
        avg_rouge = {key: np.mean(values) for key, values in rouge_scores.items()}
        return avg_rouge
    
    def calculate_clip_score(self, images, captions):
        """Calculate CLIPScore for image-text alignment"""
        clip_scores = []
        
        with torch.no_grad():
            # Process in batches
            for img, cap in zip(images, captions):
                inputs = self.clip_processor(
                    text=[cap],
                    images=img,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                clip_scores.append(logits_per_image.item())
        
        return np.mean(clip_scores)
    
    def calculate_perplexity(self):
        """Calculate perplexity on test set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Calculating perplexity"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item() * labels.numel()
                total_tokens += (labels != -100).sum().item()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def generate_and_evaluate(self, num_samples=None):
        """Generate captions and evaluate them"""
        self.model.eval()
        
        references = []
        hypotheses = []
        images = []
        
        print("Generating captions...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader)):
                if num_samples and batch_idx * config.batch_size >= num_samples:
                    break
                
                pixel_values = batch['pixel_values'].to(self.device)
                true_captions = batch['captions']
                
                # Generate captions
                generated_captions = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                )
                
                references.extend(true_captions)
                hypotheses.extend(generated_captions)
                images.extend(batch['pixel_values'])
        
        # Calculate metrics
        print("\nCalculating metrics...")
        
        # BLEU
        bleu_scores = self.calculate_bleu(references, hypotheses)
        print("\nBLEU Scores:")
        for key, value in bleu_scores.items():
            print(f"  {key}: {value:.4f}")
        
        # ROUGE
        rouge_scores = self.calculate_rouge(references, hypotheses)
        print("\nROUGE Scores:")
        for key, value in rouge_scores.items():
            if key.endswith('_f'):
                print(f"  {key}: {value:.4f}")
        
        # CLIPScore
        clip_score = self.calculate_clip_score(images[:100], hypotheses[:100])  # Sample 100 for speed
        print(f"\nCLIPScore: {clip_score:.4f}")
        
        # Perplexity
        perplexity = self.calculate_perplexity()
        print(f"Perplexity: {perplexity:.4f}")
        
        # Store results
        self.results = {
            'bleu': bleu_scores,
            'rouge': rouge_scores,
            'clip_score': clip_score,
            'perplexity': perplexity,
            'num_samples': len(references),
            'examples': []
        }
        
        # Store some examples
        for i in range(min(10, len(references))):
            self.results['examples'].append({
                'reference': references[i],
                'hypothesis': hypotheses[i]
            })
        
        return self.results
    
    def save_results(self, save_path=None):
        """Save evaluation results"""
        if save_path is None:
            save_path = os.path.join(config.results_dir, 'evaluation_results.json')
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {save_path}")
    
    def plot_results(self):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # BLEU scores
        bleu_scores = self.results['bleu']
        axes[0, 0].bar(bleu_scores.keys(), bleu_scores.values())
        axes[0, 0].set_title('BLEU Scores')
        axes[0, 0].set_ylabel('Score')
        
        # ROUGE scores (F-measure)
        rouge_f_scores = {k: v for k, v in self.results['rouge'].items() if k.endswith('_f')}
        axes[0, 1].bar(rouge_f_scores.keys(), rouge_f_scores.values())
        axes[0, 1].set_title('ROUGE F-Scores')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # CLIPScore
        axes[1, 0].bar(['CLIPScore'], [self.results['clip_score']])
        axes[1, 0].set_title('CLIPScore')
        axes[1, 0].set_ylabel('Score')
        
        # Perplexity
        axes[1, 1].bar(['Perplexity'], [self.results['perplexity']])
        axes[1, 1].set_title('Perplexity')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plot_path = os.path.join(config.results_dir, 'evaluation_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {plot_path}")


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("MemeCrafter Evaluation")
    print("=" * 60)
    
    # Set device
    device = torch.device(config.device)
    print(f"\nUsing device: {device}")
    
    # Load test data
    print("\nLoading test data...")
    test_csv = os.path.join(config.processed_data_dir, 'test.csv')
    
    # Load model
    print("\nLoading trained model...")
    model_path = os.path.join(config.models_dir, "best_model")
    model = MemeCrafterModel.load_model(model_path)
    model.to(device)
    
    # Create dataset and loader
    test_dataset = MemeDataset(test_csv, model.processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize evaluator
    evaluator = MemeEvaluator(model, test_loader, device)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.generate_and_evaluate()
    
    # Save results
    evaluator.save_results()
    
    # Plot results
    evaluator.plot_results()
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
