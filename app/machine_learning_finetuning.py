"""
Machine Learning Fine-Tuning System
===================================
Advanced fine-tuning system for continuous model improvement using feedback data.
"""

import logging
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import tempfile
from pathlib import Path
import asyncio
import aiofiles
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    model_name: str = "openai/whisper-base"
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_length: int = 448
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "models/finetuned"
    cache_dir: str = "cache"

@dataclass
class TrainingExample:
    """Training example for fine-tuning."""
    audio_path: str
    text: str
    speaker_id: str = None
    domain: str = None
    confidence: float = 1.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class FineTuningMetrics:
    """Metrics for fine-tuning evaluation."""
    train_loss: float
    eval_loss: float
    wer: float
    cer: float
    bleu: float
    rouge: float
    accuracy: float
    epoch: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class TranscriptionDataset(Dataset):
    """Custom dataset for transcription fine-tuning."""
    
    def __init__(self, examples: List[TrainingExample], processor, max_length: int = 448):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Load audio
        audio, sr = librosa.load(example.audio_path, sr=16000)
        
        # Process audio
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True,
            max_length=self.max_length
        )
        
        # Process text
        labels = self.processor.tokenizer(
            example.text,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.input_ids.squeeze(0)
        }

class MachineLearningFineTuner:
    """Advanced machine learning fine-tuning system."""
    
    def __init__(self, config: FineTuningConfig = None):
        self.config = config or FineTuningConfig()
        self.training_examples: List[TrainingExample] = []
        self.finetuned_models: Dict[str, str] = {}  # model_name -> path
        self.training_history: List[FineTuningMetrics] = []
        
        # Initialize processor and model
        self.processor = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model and processor."""
        try:
            self.processor = WhisperProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            logger.info(f"Initialized model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.processor = None
            self.model = None
    
    def add_training_examples(self, examples: List[TrainingExample]):
        """Add training examples for fine-tuning."""
        try:
            self.training_examples.extend(examples)
            logger.info(f"Added {len(examples)} training examples. Total: {len(self.training_examples)}")
            
        except Exception as e:
            logger.error(f"Failed to add training examples: {e}")
    
    def add_feedback_data(self, feedback_data: List[Dict]):
        """Add feedback data from the meta-recursive feedback system."""
        try:
            examples = []
            
            for data in feedback_data:
                if 'audio_path' in data and 'expected_text' in data:
                    example = TrainingExample(
                        audio_path=data['audio_path'],
                        text=data['expected_text'],
                        speaker_id=data.get('speaker_id'),
                        domain=data.get('domain'),
                        confidence=data.get('confidence', 1.0)
                    )
                    examples.append(example)
            
            self.add_training_examples(examples)
            logger.info(f"Added {len(examples)} examples from feedback data")
            
        except Exception as e:
            logger.error(f"Failed to add feedback data: {e}")
    
    async def start_fine_tuning(self, model_name: str = None) -> str:
        """Start fine-tuning process."""
        try:
            if not self.model or not self.processor:
                raise ValueError("Model not initialized")
            
            if len(self.training_examples) < 10:
                raise ValueError("Insufficient training examples")
            
            # Generate unique model name
            if not model_name:
                model_name = f"whisper-finetuned-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting fine-tuning for model: {model_name}")
            
            # Prepare training data
            train_examples, eval_examples = self._split_training_data()
            
            # Create datasets
            train_dataset = TranscriptionDataset(train_examples, self.processor, self.config.max_length)
            eval_dataset = TranscriptionDataset(eval_examples, self.processor, self.config.max_length)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.config.output_dir, model_name),
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,  # Disable wandb/tensorboard
                remove_unused_columns=False,
            )
            
            # Set up data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.processor.tokenizer,
                model=self.model,
                padding=True
            )
            
            # Set up trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics,
            )
            
            # Start training
            logger.info("Starting training...")
            training_result = trainer.train()
            
            # Save the fine-tuned model
            model_path = os.path.join(self.config.output_dir, model_name)
            trainer.save_model(model_path)
            self.processor.save_pretrained(model_path)
            
            self.finetuned_models[model_name] = model_path
            
            # Record training metrics
            metrics = FineTuningMetrics(
                train_loss=training_result.training_loss,
                eval_loss=training_result.eval_loss if hasattr(training_result, 'eval_loss') else 0.0,
                wer=0.0,  # Will be calculated separately
                cer=0.0,
                bleu=0.0,
                rouge=0.0,
                accuracy=0.0,
                epoch=self.config.num_epochs
            )
            self.training_history.append(metrics)
            
            logger.info(f"Fine-tuning completed. Model saved to: {model_path}")
            return model_name
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise
    
    def _split_training_data(self, train_ratio: float = 0.8) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Split training data into train and eval sets."""
        try:
            # Shuffle examples
            import random
            shuffled_examples = self.training_examples.copy()
            random.shuffle(shuffled_examples)
            
            # Split
            split_idx = int(len(shuffled_examples) * train_ratio)
            train_examples = shuffled_examples[:split_idx]
            eval_examples = shuffled_examples[split_idx:]
            
            logger.info(f"Split data: {len(train_examples)} train, {len(eval_examples)} eval")
            return train_examples, eval_examples
            
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            return self.training_examples, []
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        try:
            predictions, labels = eval_pred
            
            # Decode predictions
            pred_str = self.processor.batch_decode(predictions, skip_special_tokens=True)
            label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
            
            # Calculate metrics
            wer = self._calculate_wer(pred_str, label_str)
            cer = self._calculate_cer(pred_str, label_str)
            bleu = self._calculate_bleu(pred_str, label_str)
            
            return {
                "wer": wer,
                "cer": cer,
                "bleu": bleu
            }
            
        except Exception as e:
            logger.error(f"Metrics computation failed: {e}")
            return {"wer": 1.0, "cer": 1.0, "bleu": 0.0}
    
    def _calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate."""
        try:
            total_wer = 0.0
            count = 0
            
            for pred, ref in zip(predictions, references):
                if ref.strip():
                    wer = self._compute_wer_single(pred, ref)
                    total_wer += wer
                    count += 1
            
            return total_wer / count if count > 0 else 1.0
            
        except Exception as e:
            logger.error(f"WER calculation failed: {e}")
            return 1.0
    
    def _compute_wer_single(self, prediction: str, reference: str) -> float:
        """Compute WER for a single prediction-reference pair."""
        try:
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            if not ref_words:
                return 1.0 if pred_words else 0.0
            
            # Dynamic programming for edit distance
            d = np.zeros((len(ref_words) + 1, len(pred_words) + 1))
            
            for i in range(len(ref_words) + 1):
                d[i][0] = i
            for j in range(len(pred_words) + 1):
                d[0][j] = j
            
            for i in range(1, len(ref_words) + 1):
                for j in range(1, len(pred_words) + 1):
                    if ref_words[i-1] == pred_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
            
            return d[len(ref_words)][len(pred_words)] / len(ref_words)
            
        except Exception as e:
            logger.error(f"Single WER computation failed: {e}")
            return 1.0
    
    def _calculate_cer(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Character Error Rate."""
        try:
            total_cer = 0.0
            count = 0
            
            for pred, ref in zip(predictions, references):
                if ref.strip():
                    cer = self._compute_cer_single(pred, ref)
                    total_cer += cer
                    count += 1
            
            return total_cer / count if count > 0 else 1.0
            
        except Exception as e:
            logger.error(f"CER calculation failed: {e}")
            return 1.0
    
    def _compute_cer_single(self, prediction: str, reference: str) -> float:
        """Compute CER for a single prediction-reference pair."""
        try:
            pred_chars = list(prediction.lower())
            ref_chars = list(reference.lower())
            
            if not ref_chars:
                return 1.0 if pred_chars else 0.0
            
            # Dynamic programming for edit distance
            d = np.zeros((len(ref_chars) + 1, len(pred_chars) + 1))
            
            for i in range(len(ref_chars) + 1):
                d[i][0] = i
            for j in range(len(pred_chars) + 1):
                d[0][j] = j
            
            for i in range(1, len(ref_chars) + 1):
                for j in range(1, len(pred_chars) + 1):
                    if ref_chars[i-1] == pred_chars[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
            
            return d[len(ref_chars)][len(pred_chars)] / len(ref_chars)
            
        except Exception as e:
            logger.error(f"Single CER computation failed: {e}")
            return 1.0
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            total_bleu = 0.0
            count = 0
            smoothing = SmoothingFunction().method1
            
            for pred, ref in zip(predictions, references):
                if ref.strip():
                    ref_tokens = ref.lower().split()
                    pred_tokens = pred.lower().split()
                    
                    if ref_tokens and pred_tokens:
                        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
                        total_bleu += bleu
                        count += 1
            
            return total_bleu / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0
    
    async def evaluate_model(self, model_name: str, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Evaluate a fine-tuned model."""
        try:
            if model_name not in self.finetuned_models:
                raise ValueError(f"Model {model_name} not found")
            
            model_path = self.finetuned_models[model_name]
            
            # Load the fine-tuned model
            processor = WhisperProcessor.from_pretrained(model_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
            
            # Evaluate on test examples
            predictions = []
            references = []
            
            for example in test_examples:
                try:
                    # Load and process audio
                    audio, sr = librosa.load(example.audio_path, sr=16000)
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                    
                    # Generate prediction
                    with torch.no_grad():
                        generated_ids = model.generate(
                            inputs.input_features,
                            max_length=448,
                            num_beams=5,
                            early_stopping=True
                        )
                    
                    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    predictions.append(prediction)
                    references.append(example.text)
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed for example {example.audio_path}: {e}")
                    predictions.append("")
                    references.append(example.text)
            
            # Calculate metrics
            metrics = {
                'wer': self._calculate_wer(predictions, references),
                'cer': self._calculate_cer(predictions, references),
                'bleu': self._calculate_bleu(predictions, references)
            }
            
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
    
    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return [asdict(metric) for metric in self.training_history]
    
    def get_available_models(self) -> List[str]:
        """Get list of available fine-tuned models."""
        return list(self.finetuned_models.keys())
    
    def load_model(self, model_name: str) -> bool:
        """Load a fine-tuned model for inference."""
        try:
            if model_name not in self.finetuned_models:
                logger.error(f"Model {model_name} not found")
                return False
            
            model_path = self.finetuned_models[model_name]
            
            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            
            logger.info(f"Loaded fine-tuned model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    async def transcribe_with_finetuned_model(self, audio_path: str, model_name: str = None) -> str:
        """Transcribe audio using a fine-tuned model."""
        try:
            if not self.model or not self.processor:
                if model_name and not self.load_model(model_name):
                    raise ValueError(f"Failed to load model: {model_name}")
                elif not model_name:
                    raise ValueError("No model loaded")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    max_length=448,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return transcription
            
        except Exception as e:
            logger.error(f"Fine-tuned transcription failed: {e}")
            return ""
    
    def export_training_data(self, output_path: str) -> bool:
        """Export training data for external use."""
        try:
            data = {
                'examples': [asdict(example) for example in self.training_examples],
                'config': asdict(self.config),
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Training data exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Training data export failed: {e}")
            return False
    
    def import_training_data(self, input_path: str) -> bool:
        """Import training data from external source."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Import examples
            examples = []
            for example_data in data.get('examples', []):
                example = TrainingExample(
                    audio_path=example_data['audio_path'],
                    text=example_data['text'],
                    speaker_id=example_data.get('speaker_id'),
                    domain=example_data.get('domain'),
                    confidence=example_data.get('confidence', 1.0)
                )
                examples.append(example)
            
            self.add_training_examples(examples)
            
            logger.info(f"Training data imported from: {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Training data import failed: {e}")
            return False

# Global fine-tuning system instance
fine_tuning_system = MachineLearningFineTuner()

# API functions
async def start_fine_tuning(model_name: str = None) -> str:
    """Start fine-tuning process."""
    return await fine_tuning_system.start_fine_tuning(model_name)

async def add_feedback_data(feedback_data: List[Dict]):
    """Add feedback data for fine-tuning."""
    fine_tuning_system.add_feedback_data(feedback_data)

async def evaluate_finetuned_model(model_name: str, test_examples: List[Dict]) -> Dict[str, float]:
    """Evaluate a fine-tuned model."""
    examples = [TrainingExample(**data) for data in test_examples]
    return await fine_tuning_system.evaluate_model(model_name, examples)

async def transcribe_with_finetuned_model(audio_path: str, model_name: str = None) -> str:
    """Transcribe using fine-tuned model."""
    return await fine_tuning_system.transcribe_with_finetuned_model(audio_path, model_name)
