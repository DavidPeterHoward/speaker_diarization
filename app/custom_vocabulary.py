"""
Custom Vocabulary and Domain Adaptation System
=============================================
Advanced vocabulary management and domain-specific adaptation for improved transcription accuracy.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

@dataclass
class VocabularyTerm:
    """Represents a vocabulary term with metadata."""
    term: str
    pronunciation: Optional[str] = None
    boost_factor: float = 1.0
    domain: str = "general"
    frequency: int = 0
    confidence: float = 1.0
    alternatives: List[str] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []

@dataclass
class DomainProfile:
    """Represents a domain-specific vocabulary profile."""
    name: str
    terms: List[VocabularyTerm]
    language_model_weight: float = 1.0
    acoustic_model_weight: float = 1.0
    context_prompts: List[str] = None
    
    def __post_init__(self):
        if self.context_prompts is None:
            self.context_prompts = []

class CustomVocabularyManager:
    """Advanced vocabulary management system."""
    
    def __init__(self, config_path: str = "data/vocabulary_config.json"):
        self.config_path = config_path
        self.domains: Dict[str, DomainProfile] = {}
        self.terms: Dict[str, VocabularyTerm] = {}
        self.boosted_phrases: Dict[str, float] = {}
        self.context_prompts: List[str] = []
        
        # Load existing configuration
        self.load_configuration()
    
    def add_domain_vocabulary(self, domain: str, terms: List[str], 
                            pronunciations: List[str] = None, 
                            boost_factors: List[float] = None) -> bool:
        """Add domain-specific vocabulary."""
        try:
            if pronunciations is None:
                pronunciations = [None] * len(terms)
            if boost_factors is None:
                boost_factors = [1.0] * len(terms)
            
            vocabulary_terms = []
            for i, term in enumerate(terms):
                vocab_term = VocabularyTerm(
                    term=term,
                    pronunciation=pronunciations[i],
                    boost_factor=boost_factors[i],
                    domain=domain
                )
                vocabulary_terms.append(vocab_term)
                self.terms[term.lower()] = vocab_term
            
            # Create or update domain profile
            if domain in self.domains:
                self.domains[domain].terms.extend(vocabulary_terms)
            else:
                self.domains[domain] = DomainProfile(
                    name=domain,
                    terms=vocabulary_terms
                )
            
            self.save_configuration()
            logger.info(f"Added {len(terms)} terms to domain '{domain}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add domain vocabulary: {e}")
            return False
    
    def boost_phrases(self, phrases: List[str], boost_factor: float = 2.0) -> bool:
        """Boost specific phrases for better recognition."""
        try:
            for phrase in phrases:
                self.boosted_phrases[phrase.lower()] = boost_factor
                logger.info(f"Boosted phrase '{phrase}' with factor {boost_factor}")
            
            self.save_configuration()
            return True
            
        except Exception as e:
            logger.error(f"Failed to boost phrases: {e}")
            return False
    
    def add_context_prompts(self, prompts: List[str]) -> bool:
        """Add context prompts for better transcription."""
        try:
            self.context_prompts.extend(prompts)
            self.save_configuration()
            logger.info(f"Added {len(prompts)} context prompts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add context prompts: {e}")
            return False
    
    def get_enhanced_prompt(self, audio_context: str = None, domain: str = None) -> str:
        """Generate enhanced prompt with custom vocabulary."""
        try:
            prompt_parts = []
            
            # Add audio context if provided
            if audio_context:
                prompt_parts.append(f"Context: {audio_context}")
            
            # Add domain-specific vocabulary
            if domain and domain in self.domains:
                domain_terms = [term.term for term in self.domains[domain].terms]
                if domain_terms:
                    vocab_text = f"Key terms for {domain}: {', '.join(domain_terms[:20])}"  # Limit to 20 terms
                    prompt_parts.append(vocab_text)
            
            # Add boosted phrases
            if self.boosted_phrases:
                boosted_text = f"Important phrases: {', '.join(list(self.boosted_phrases.keys())[:10])}"
                prompt_parts.append(boosted_text)
            
            # Add context prompts
            if self.context_prompts:
                context_text = f"Context prompts: {'; '.join(self.context_prompts[:3])}"
                prompt_parts.append(context_text)
            
            return " ".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced prompt: {e}")
            return ""
    
    def get_vocabulary_boost_config(self, domain: str = None) -> Dict[str, Any]:
        """Get vocabulary boost configuration for transcription."""
        try:
            config = {
                'boosted_phrases': self.boosted_phrases.copy(),
                'domain_terms': {},
                'pronunciation_guide': {},
                'language_model_weights': {}
            }
            
            # Add domain-specific terms
            if domain and domain in self.domains:
                domain_profile = self.domains[domain]
                for term in domain_profile.terms:
                    config['domain_terms'][term.term] = {
                        'boost_factor': term.boost_factor,
                        'pronunciation': term.pronunciation,
                        'alternatives': term.alternatives
                    }
                    
                    if term.pronunciation:
                        config['pronunciation_guide'][term.term] = term.pronunciation
                
                config['language_model_weights'][domain] = domain_profile.language_model_weight
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to get vocabulary boost config: {e}")
            return {}
    
    def analyze_transcription_errors(self, transcription: str, reference: str = None) -> Dict[str, Any]:
        """Analyze transcription errors and suggest vocabulary improvements."""
        try:
            analysis = {
                'missing_terms': [],
                'misrecognized_terms': [],
                'suggestions': [],
                'confidence_scores': {}
            }
            
            # Extract terms from transcription
            transcription_terms = self._extract_terms(transcription)
            
            # Check for missing domain terms
            for domain, profile in self.domains.items():
                for term in profile.terms:
                    if term.term.lower() not in [t.lower() for t in transcription_terms]:
                        analysis['missing_terms'].append({
                            'term': term.term,
                            'domain': domain,
                            'suggestion': f"Consider adding pronunciation guide for '{term.term}'"
                        })
            
            # Check for misrecognized terms (if reference provided)
            if reference:
                reference_terms = self._extract_terms(reference)
                misrecognized = self._find_misrecognized_terms(transcription_terms, reference_terms)
                analysis['misrecognized_terms'] = misrecognized
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze transcription errors: {e}")
            return {}
    
    def suggest_vocabulary_improvements(self, transcription: str, domain: str = None) -> List[str]:
        """Suggest vocabulary improvements based on transcription analysis."""
        try:
            suggestions = []
            
            # Extract terms and analyze
            terms = self._extract_terms(transcription)
            term_frequencies = Counter(terms)
            
            # Suggest boosting for frequent terms
            for term, freq in term_frequencies.most_common(10):
                if freq > 2 and term.lower() not in self.boosted_phrases:
                    suggestions.append(f"Consider boosting term '{term}' (appears {freq} times)")
            
            # Suggest domain-specific additions
            if domain and domain in self.domains:
                domain_terms = [t.term.lower() for t in self.domains[domain].terms]
                missing_terms = [term for term in terms if term.lower() not in domain_terms]
                
                if missing_terms:
                    suggestions.append(f"Consider adding to {domain} vocabulary: {', '.join(missing_terms[:5])}")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest vocabulary improvements: {e}")
            return []
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text."""
        # Simple term extraction - can be enhanced with NLP
        terms = re.findall(r'\b\w+\b', text.lower())
        return terms
    
    def _find_misrecognized_terms(self, transcription_terms: List[str], reference_terms: List[str]) -> List[Dict]:
        """Find misrecognized terms by comparing transcription with reference."""
        misrecognized = []
        
        # Simple comparison - can be enhanced with fuzzy matching
        for ref_term in reference_terms:
            if ref_term not in transcription_terms:
                # Find similar terms in transcription
                similar_terms = [t for t in transcription_terms if self._similarity(ref_term, t) > 0.7]
                if similar_terms:
                    misrecognized.append({
                        'reference': ref_term,
                        'transcription': similar_terms[0],
                        'similarity': self._similarity(ref_term, similar_terms[0])
                    })
        
        return misrecognized
    
    def _similarity(self, term1: str, term2: str) -> float:
        """Calculate similarity between two terms."""
        # Simple character-based similarity
        if not term1 or not term2:
            return 0.0
        
        # Jaccard similarity
        set1 = set(term1.lower())
        set2 = set(term2.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def load_configuration(self) -> bool:
        """Load vocabulary configuration from file."""
        try:
            if not os.path.exists(self.config_path):
                logger.info("No existing vocabulary configuration found")
                return True
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Load domains
            self.domains = {}
            for domain_name, domain_data in config.get('domains', {}).items():
                terms = [VocabularyTerm(**term_data) for term_data in domain_data.get('terms', [])]
                self.domains[domain_name] = DomainProfile(
                    name=domain_name,
                    terms=terms,
                    language_model_weight=domain_data.get('language_model_weight', 1.0),
                    acoustic_model_weight=domain_data.get('acoustic_model_weight', 1.0),
                    context_prompts=domain_data.get('context_prompts', [])
                )
            
            # Load terms
            self.terms = {}
            for term_data in config.get('terms', []):
                term = VocabularyTerm(**term_data)
                self.terms[term.term.lower()] = term
            
            # Load boosted phrases
            self.boosted_phrases = config.get('boosted_phrases', {})
            
            # Load context prompts
            self.context_prompts = config.get('context_prompts', [])
            
            logger.info(f"Loaded vocabulary configuration: {len(self.domains)} domains, {len(self.terms)} terms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary configuration: {e}")
            return False
    
    def save_configuration(self) -> bool:
        """Save vocabulary configuration to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config = {
                'domains': {},
                'terms': [],
                'boosted_phrases': self.boosted_phrases,
                'context_prompts': self.context_prompts
            }
            
            # Save domains
            for domain_name, domain in self.domains.items():
                config['domains'][domain_name] = {
                    'name': domain.name,
                    'terms': [asdict(term) for term in domain.terms],
                    'language_model_weight': domain.language_model_weight,
                    'acoustic_model_weight': domain.acoustic_model_weight,
                    'context_prompts': domain.context_prompts
                }
            
            # Save terms
            config['terms'] = [asdict(term) for term in self.terms.values()]
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info("Vocabulary configuration saved")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vocabulary configuration: {e}")
            return False

# Global vocabulary manager instance
vocabulary_manager = CustomVocabularyManager()

# Plugin registration
def custom_vocabulary_enhancement(segments: List) -> List:
    """Custom vocabulary enhancement plugin."""
    try:
        # This would be called during post-processing
        # Implementation depends on specific transcription backend
        logger.info("Applied custom vocabulary enhancement")
        return segments
    except Exception as e:
        logger.error(f"Custom vocabulary enhancement failed: {e}")
        return segments
