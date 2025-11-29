# ============================================================================
# OPTIMIZED MULTI-VERSION RAG SYSTEM - WITH GROUND TRUTH EVALUATION
# ============================================================================

from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
import json
import os
from dotenv import load_dotenv
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher
import pickle
import hashlib

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

load_dotenv()

# ============================================================================
# EMBEDDING CACHE MANAGER - SAVE/LOAD EMBEDDINGS
# ============================================================================

class EmbeddingCache:
    """Cache embeddings to disk to avoid regenerating"""
    
    def __init__(self, cache_dir: str = "embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, json_file: str, embedding_model: str) -> str:
        """Generate unique cache key based on file content and model"""
        with open(json_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{file_hash}_{embedding_model.replace('/', '_')}"
    
    def load_vector_store(self, json_file: str, embedding_model: str) -> Optional[FAISS]:
        """Load cached FAISS vector store"""
        cache_key = self._get_cache_key(json_file, embedding_model)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}_faiss")
        
        if os.path.exists(cache_path):
            try:
                embeddings = HuggingFaceEndpointEmbeddings(
                    repo_id="BAAI/bge-small-en-v1.5",
                    task="feature-extraction",
                    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
                )
                vector_store = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
                print(f"      ✅ Loaded cached embeddings from {cache_path}")
                return vector_store
            except Exception as e:
                print(f"      ⚠️  Failed to load cache: {e}")
                return None
        return None
    
    def save_vector_store(self, vector_store: FAISS, json_file: str, embedding_model: str):
        """Save FAISS vector store to cache"""
        cache_key = self._get_cache_key(json_file, embedding_model)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}_faiss")
        
        try:
            vector_store.save_local(cache_path)
            print(f"      ✅ Saved embeddings to cache: {cache_path}")
        except Exception as e:
            print(f"      ⚠️  Failed to save cache: {e}")
    
    def load_bm25(self, json_file: str) -> Optional[tuple]:
        """Load cached BM25 index"""
        cache_key = self._get_cache_key(json_file, "bm25")
        cache_path = os.path.join(self.cache_dir, f"{cache_key}_bm25.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"      ✅ Loaded cached BM25 index")
                return data['bm25'], data['corpus'], data['tokenized_corpus']
            except Exception as e:
                print(f"      ⚠️  Failed to load BM25 cache: {e}")
                return None
        return None
    
    def save_bm25(self, bm25, corpus, tokenized_corpus, json_file: str):
        """Save BM25 index to cache"""
        cache_key = self._get_cache_key(json_file, "bm25")
        cache_path = os.path.join(self.cache_dir, f"{cache_key}_bm25.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'bm25': bm25,
                    'corpus': corpus,
                    'tokenized_corpus': tokenized_corpus
                }, f)
            print(f"      ✅ Saved BM25 index to cache")
        except Exception as e:
            print(f"      ⚠️  Failed to save BM25 cache: {e}")

# ============================================================================
# LANGUAGE HANDLER (NO LLM CALLS)
# ============================================================================

class LanguageHandler:
    """Handle bilingual queries (English/Hindi) - NOW WITH LLM TRANSLATION"""
    
    def __init__(self, model=None):
        self.model = model  # Add model to constructor
    
    @staticmethod
    def detect_language(text: str) -> str:
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        return 'hindi' if hindi_chars > len(text) * 0.3 else 'english'
    
    def translate_query(self, query: str, target_lang: str = 'english') -> str:
        """Translate Hindi to English using Gemini LLM"""
        
        # If no model provided or already English, return as-is
        if not self.model or target_lang != 'english':
            return query
        
        # Check if query is in Hindi
        if self.detect_language(query) != 'hindi':
            return query
        
        # Use Gemini for translation
        try:
            prompt = f"""Translate this Hindi text to English. Only provide the translation, nothing else.

Hindi: {query}
English:"""
            
            response = self.model.invoke(prompt)
            translated = response.content.strip()
            print(f"      🔄 Translated: '{query}' -> '{translated}'")
            return translated
            
        except Exception as e:
            print(f"      ⚠️ Translation failed: {e}, using original query")
            return query
    

class QueryRelevanceChecker:
    """Check if query is related to retail/products - USING LLM"""
    
    def __init__(self, model):
        self.model = model
    
    def is_retail_related(self, query: str) -> tuple:
        """
        Check if query is retail-related using LLM
        Returns: (is_relevant: bool, explanation: str)
        """
        
        prompt = f"""You are a query classifier for a retail product assistant.

Your task: Determine if the following query is related to retail, products, shopping, inventory, or store operations.

Query: "{query}"

Respond with ONLY ONE WORD:
- "YES" if the query is about products, prices, locations, recommendations, inventory, shopping, or store-related topics
- "NO" if the query is about anything else (politics, weather, celebrities, general knowledge, math, coding, etc.)

Answer (YES/NO):"""

        try:
            response = self.model.invoke(prompt)
            answer = response.content.strip().upper()
            
            # Parse response
            if "YES" in answer[:10]:  # Check first 10 chars for "YES"
                return True, "Query is retail-related"
            else:
                return False, "Query is not retail-related"
                
        except Exception as e:
            # If LLM fails, default to rejecting (safer)
            print(f"⚠️ Relevance check failed: {e}")
            return False, "Unable to verify query relevance"

# ============================================================================
# VERSION 1: BASIC RAG (OPTIMIZED)
# ============================================================================

class BasicRAG:
    """Version 1: Simple vector search - OPTIMIZED"""
    
    def __init__(self, vector_store: FAISS, model):
        self.vector_store = vector_store
        self.model = model
        self.lang_handler = LanguageHandler(model)
        
    def query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        lang = self.lang_handler.detect_language(query)
        if lang == 'hindi':
            query_en = self.lang_handler.translate_query(query, 'english')
        else:
            query_en = query
        
        results = self.vector_store.similarity_search(query_en, k=3)
        context = "\n".join([r.page_content[:300] for r in results])
        retrieved_docs = [r.page_content for r in results]
        # print(f"DEBUG - Retrieved context for '{query}': {context[:200]}...")
        
        prompt = f"""Q: {query}
Context: {context}n


Brief answer (if location: "Location: <tray>, Row: <row>, Col: <col>"):"""
        
        response = self.model.invoke(prompt)
        latency = time.time() - start_time
        
        return {
            'response': response.content,
            'latency': latency,
            'context': context,
            'retrieved_docs': retrieved_docs,
            'num_docs': len(results)
        }

# ============================================================================
# VERSION 2: HYBRID RETRIEVAL + RERANKING (OPTIMIZED)
# ============================================================================

class HybridRAG:
    """Version 2: Dense + Sparse + Reranking - OPTIMIZED"""
    
    def __init__(self, vector_store: FAISS, bm25_data: tuple, model):
        self.vector_store = vector_store
        self.model = model
        self.lang_handler = LanguageHandler(model)
        
        self.bm25, self.corpus, self.tokenized_corpus = bm25_data
        
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            self.reranker = None
            print("⚠️  Cross-encoder not available, skipping reranking")
    
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def _query_expansion_simple(self, query: str) -> List[str]:
        words = query.lower().split()
        expanded = [query]
        
        if any(w.endswith('s') for w in words):
            expanded.append(' '.join([w.rstrip('s') if w.endswith('s') else w for w in words]))
        
        return expanded[:2]
    
    def hybrid_retrieve(self, query: str, k: int = 5) -> List[Dict]:
        queries = self._query_expansion_simple(query)
        
        all_results = {}
        
        for q in queries:
            vector_results = self.vector_store.similarity_search_with_score(q, k=k)
            
            tokenized_query = self._tokenize(q)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]
            
            for idx, (doc, score) in enumerate(vector_results):
                doc_key = doc.page_content
                if doc_key not in all_results:
                    all_results[doc_key] = {'doc': doc_key, 'vector_rank': None, 'bm25_rank': None}
                all_results[doc_key]['vector_rank'] = idx + 1
            
            for rank, idx in enumerate(bm25_top_indices):
                doc_key = json.dumps(self.corpus[idx])
                if doc_key not in all_results:
                    all_results[doc_key] = {'doc': doc_key, 'vector_rank': None, 'bm25_rank': None}
                all_results[doc_key]['bm25_rank'] = rank + 1
        
        for result in all_results.values():
            rrf_score = 0
            if result['vector_rank']:
                rrf_score += 1 / (60 + result['vector_rank'])
            if result['bm25_rank']:
                rrf_score += 1 / (60 + result['bm25_rank'])
            result['rrf_score'] = rrf_score
        
        sorted_results = sorted(all_results.values(), key=lambda x: x['rrf_score'], reverse=True)
        return sorted_results[:k]
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 3) -> List[str]:
        if not self.reranker:
            return [r['doc'] for r in results[:top_k]]
        
        pairs = [[query, r['doc'][:500]] for r in results]
        scores = self.reranker.predict(pairs)
        ranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [r[0]['doc'] for r in ranked_results[:top_k]]
    
    def query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        lang = self.lang_handler.detect_language(query)
        if lang == 'hindi':
            query_en = self.lang_handler.translate_query(query, 'english')
        else:
            query_en = query
        
        results = self.hybrid_retrieve(query_en, k=5)
        top_docs = self.rerank(query_en, results, top_k=3)
        context = "\n".join([doc[:300] for doc in top_docs])
        
        prompt = f"""Q: {query}
Context: {context}

Brief answer:"""
        
        response = self.model.invoke(prompt)
        latency = time.time() - start_time
        
        return {
            'response': response.content,
            'latency': latency,
            'context': context,
            'retrieved_docs': top_docs,
            'num_docs': len(top_docs),
            'num_queries_expanded': len(self._query_expansion_simple(query_en))
        }

# ============================================================================
# VERSION 3: AGENTIC RAG (OPTIMIZED - REDUCED ITERATIONS)
# ============================================================================

class ProductTools:
    """Tools for product operations - NO EXTRA LLM CALLS"""
    
    def __init__(self, products_data, vector_store):
        self.products = products_data
        self.vector_store = vector_store
    
    def search_by_name(self, name: str) -> List[Dict]:
        results = []
        for p in self.products:
            if name.lower() in p.get('name', '').lower():
                results.append(p)
        return results[:3]
    
    def search_by_category(self, category: str) -> List[Dict]:
        results = []
        for p in self.products:
            if category.lower() in p.get('category', '').lower():
                results.append(p)
        return results[:3]
    
    def get_location(self, product_name: str) -> Optional[Dict]:
        for p in self.products:
            if product_name.lower() in p.get('name', '').lower():
                return {
                    'name': p.get('name'),
                    'tray': p.get('tray'),
                    'row': p.get('row'),
                    'col': p.get('col')
                }
        return None
    
    def recommend_similar(self, query: str, k: int = 3) -> List[Dict]:
        results = self.vector_store.similarity_search(query, k=k)
        return [json.loads(r.page_content) for r in results]


class AgenticRAG:
    """Version 3: ReACT-style agent - OPTIMIZED (2 iterations max)"""
    
    def __init__(self, vector_store: FAISS, products_data, model):
        self.vector_store = vector_store
        self.tools = ProductTools(products_data, vector_store)
        self.model = model
        self.lang_handler = LanguageHandler(model)
        
    def parse_tool_call(self, thought: str) -> Optional[Dict]:
        tool_patterns = {
            'search_by_name': r'search_by_name\("([^"]+)"\)',
            'search_by_category': r'search_by_category\("([^"]+)"\)',
            'get_location': r'get_location\("([^"]+)"\)',
            'recommend_similar': r'recommend_similar\("([^"]+)"(?:,\s*(\d+))?\)',
        }
        
        for tool, pattern in tool_patterns.items():
            match = re.search(pattern, thought)
            if match:
                args = match.groups()
                return {'tool': tool, 'args': [a for a in args if a]}
        return None
    
    def execute_tool(self, tool_call: Dict) -> Any:
        tool = tool_call['tool']
        args = tool_call['args']
        
        if tool == 'search_by_name':
            return self.tools.search_by_name(args[0])
        elif tool == 'search_by_category':
            return self.tools.search_by_category(args[0])
        elif tool == 'get_location':
            return self.tools.get_location(args[0])
        elif tool == 'recommend_similar':
            k = int(args[1]) if len(args) > 1 else 3
            return self.tools.recommend_similar(args[0], k)
        return None
    
    def query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        lang = self.lang_handler.detect_language(query)
        if lang == 'hindi':
            query_en = self.lang_handler.translate_query(query, 'english')
        else:
            query_en = query
        
        system_prompt = """You are a retail assistant. Tools:
- search_by_name("name")
- get_location("name")
- recommend_similar("query", k)

Format: Thought -> Action -> Answer
If location: "Location: <tray>, Row: <row>, Col: <col>" """
        
        conversation = f"{system_prompt}\n\nQ: {query}\n\n"
        
        thoughts = []
        actions = []
        retrieved_docs = []
        
        for i in range(2):
            response = self.model.invoke(conversation)
            thought = response.content
            thoughts.append(thought)
            
            if 'Answer:' in thought:
                answer = thought.split('Answer:')[-1].strip()
                latency = time.time() - start_time
                context = '\n'.join(thoughts)
                return {
                    'response': answer,
                    'latency': latency,
                    'context': context,
                    'retrieved_docs': retrieved_docs,
                    'num_iterations': i + 1,
                    'num_tool_calls': len(actions)
                }
            
            tool_call = self.parse_tool_call(thought)
            if tool_call:
                actions.append(tool_call)
                observation = self.execute_tool(tool_call)
                retrieved_docs.append(json.dumps(observation)[:500])
                conversation += f"{thought}\nObs: {json.dumps(observation)[:500]}\n\n"
            else:
                conversation += f"{thought}\n\n"
        
        latency = time.time() - start_time
        context = '\n'.join(thoughts)
        return {
            'response': thoughts[-1] if thoughts else "Unable to answer",
            'latency': latency,
            'context': context,
            'retrieved_docs': retrieved_docs,
            'num_iterations': 2,
            'num_tool_calls': len(actions)
        }

# ============================================================================
# VERSION 4: CORRECTIVE RAG (OPTIMIZED - MINIMAL GRADING)
# ============================================================================

class CorrectiveRAG:
    """Version 4: CRAG - OPTIMIZED (less grading calls)"""
    
    def __init__(self, vector_store: FAISS, products_data, model):
        self.vector_store = vector_store
        self.tools = ProductTools(products_data, vector_store)
        self.model = model
        self.lang_handler = LanguageHandler(model)
    
    def grade_relevance_simple(self, query: str, docs: List[str]) -> List[Dict]:
        query_words = set(query.lower().split())
        graded_docs = []
        
        for doc in docs[:3]:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0
            graded_docs.append({'doc': doc, 'relevance_score': score})
        
        return sorted(graded_docs, key=lambda x: x['relevance_score'], reverse=True)
    
    def query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        lang = self.lang_handler.detect_language(query)
        if lang == 'hindi':
            query_en = self.lang_handler.translate_query(query, 'english')
        else:
            query_en = query
        
        results = self.vector_store.similarity_search(query_en, k=3)
        docs = [r.page_content for r in results]
        
        graded = self.grade_relevance_simple(query_en, docs)
        avg_relevance = np.mean([g['relevance_score'] for g in graded])
        
        if avg_relevance < 0.3:
            similar_products = self.tools.recommend_similar(query_en, k=3)
            context = json.dumps(similar_products)[:500]
            retrieved_docs = [json.dumps(p)[:300] for p in similar_products]
        else:
            top_docs = [g['doc'][:300] for g in graded[:2]]
            context = "\n".join(top_docs)
            retrieved_docs = top_docs
        
        prompt = f"""Q: {query}
Context: {context}

Brief answer:"""
        
        response = self.model.invoke(prompt)
        
        latency = time.time() - start_time
        
        return {
            'response': response.content,
            'latency': latency,
            'context': context,
            'retrieved_docs': retrieved_docs,
            'avg_relevance': avg_relevance,
            'quality_score': 0.8
        }

# ============================================================================
# EVALUATOR WITH GROUND TRUTH SUPPORT
# ============================================================================

class RAGEvaluator:
    """Evaluator with ground truth support"""
    
    def __init__(self, model=None):
        self.model = model
    
    def context_relevance(self, query: str, context: str) -> float:
        """Context Relevance: Measures how relevant the retrieved context is to the query"""
        if not context or not query:
            return 0.0
        
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                      'this', 'that', 'with', 'from', 'have', 'has', 'will', 
                      'can', 'your', 'more', 'what', 'where', 'when', 'how'}
        
        def get_terms(text):
            words = re.findall(r'\w+', text.lower())
            return [w for w in words if len(w) > 2 and w not in stop_words]
        
        query_terms = get_terms(query)
        context_terms = get_terms(context)
        
        if not query_terms or not context_terms:
            return 0.0
        
        query_freq = Counter(query_terms)
        context_freq = Counter(context_terms)
        
        common_terms = set(query_terms) & set(context_terms)
        
        if not common_terms:
            return 0.0
        
        numerator = sum(query_freq[term] * context_freq[term] for term in common_terms)
        
        query_magnitude = np.sqrt(sum(freq ** 2 for freq in query_freq.values()))
        context_magnitude = np.sqrt(sum(freq ** 2 for freq in context_freq.values()))
        
        if query_magnitude == 0 or context_magnitude == 0:
            return 0.0
        
        cosine_sim = numerator / (query_magnitude * context_magnitude)
        
        return min(1.0, cosine_sim)
    
    def bleu_score(self, prediction: str, reference: str) -> float:
        """BLEU Score - compares prediction to ground truth"""
        def get_ngrams(text: str, n: int = 2) -> Counter:
            words = text.lower().split()
            return Counter([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
        
        pred_bigrams = get_ngrams(prediction, 2)
        ref_bigrams = get_ngrams(reference, 2)
        
        if not pred_bigrams or not ref_bigrams:
            return 0.0
        
        matches = sum((pred_bigrams & ref_bigrams).values())
        total = sum(pred_bigrams.values())
        
        bleu = matches / total if total > 0 else 0.0
        
        pred_len = len(prediction.split())
        ref_len = len(reference.split())
        if pred_len < ref_len:
            bp = np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0
        else:
            bp = 1.0
        
        return bleu * bp
    
    def rouge_score(self, prediction: str, reference: str) -> float:
        """ROUGE-L - compares prediction to ground truth"""
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()
        
        if not pred_words or not ref_words:
            return 0.0
        
        matcher = SequenceMatcher(None, pred_words, ref_words)
        lcs_length = sum(block.size for block in matcher.get_matching_blocks())
        
        if len(pred_words) == 0 or len(ref_words) == 0:
            return 0.0
        
        r_lcs = lcs_length / len(ref_words)
        p_lcs = lcs_length / len(pred_words)
        
        if r_lcs + p_lcs == 0:
            return 0.0
        
        f_lcs = (2 * r_lcs * p_lcs) / (r_lcs + p_lcs)
        return f_lcs
    
    def hallucination_rate(self, response: str, context: str) -> float:
        """Hallucination rate - checks if response claims are supported by context"""
        if not context or not response:
            return 1.0
        
        response_lower = response.lower()
        context_lower = context.lower()
        
        response_numbers = set(re.findall(r'\d+', response_lower))
        context_numbers = set(re.findall(r'\d+', context_lower))
        
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'this', 'that', 
                      'with', 'from', 'have', 'has', 'will', 'can', 'your', 'more'}
        
        response_terms = set()
        for word in response_lower.split():
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 3 and word_clean not in stop_words:
                response_terms.add(word_clean)
        
        context_terms = set()
        for word in context_lower.split():
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 3 and word_clean not in stop_words:
                context_terms.add(word_clean)
        
        unsupported_numbers = response_numbers - context_numbers
        unsupported_terms = response_terms - context_terms
        
        total_facts = len(response_numbers) + len(response_terms)
        if total_facts == 0:
            return 0.0
        
        unsupported_facts = len(unsupported_numbers) + min(len(unsupported_terms), len(response_terms) // 2)
        
        hallucination_rate = unsupported_facts / total_facts
        
        return max(0.0, min(1.0, hallucination_rate))
    
    def evaluate(self, query: str, response: str, context: str, 
                 retrieved_docs: List[str] = None, ground_truth: str = None) -> Dict[str, float]:
        """Calculate all metrics - use ground truth if available"""
        
        if retrieved_docs is None:
            retrieved_docs = context.split('\n') if context else []
        
        metrics = {
            'context_relevance': self.context_relevance(query, context),
            'hallucination_rate': self.hallucination_rate(response, context),
        }
        
        # Use ground truth if available, otherwise use query as reference
        reference = ground_truth if ground_truth else query
        metrics['bleu'] = self.bleu_score(response, reference)
        metrics['rouge'] = self.rouge_score(response, reference)
        
        metrics['overall_score'] = (
            metrics['context_relevance'] * 0.25 +
            metrics['bleu'] * 0.25 +
            metrics['rouge'] * 0.30 +
            (1 - metrics['hallucination_rate']) * 0.20
        )
        
        return metrics

# ============================================================================
# RESULTS STORAGE & VISUALIZATION
# ============================================================================

class ResultsManager:
    """Store and visualize results"""
    
    def __init__(self):
        self.history = []
        self.results_file = 'rag_comparison_results.csv'
        self.charts_dir = 'rag_charts'
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def add_result(self, query: str, version_results: Dict, ground_truth: str = None, context: str = None):
        """Store result for a query"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            'timestamp': timestamp,
            'query': query,
            'ground_truth': ground_truth,
            'context': context,  # Add this line
            **version_results
        }
        
        self.history.append(result)
        self._save_to_csv()
    
    def _save_to_csv(self):
        """Save results to CSV"""
        if not self.history:
            return
        
        flattened = []
        for result in self.history:
            flat = {
                'timestamp': result['timestamp'], 
                'query': result['query'],
                'ground_truth': result.get('ground_truth', ''),
                'context': result.get('context', '')  # Add this line
            }
            
            for version, data in result.items():
                if version not in ['timestamp', 'query', 'ground_truth', 'context']:  # Add 'context' here
                    for key, value in data.items():
                        flat[f'{version}_{key}'] = value
            
            flattened.append(flat)
        
        df = pd.DataFrame(flattened)
        df.to_csv(self.results_file, index=False)
        print(f"\n✅ Results saved to {self.results_file}")
    
    def plot_comparison(self, query_idx: int = -1):
        """Create comparison chart"""
        if not self.history:
            return
        
        result = self.history[query_idx]
        query = result['query']
        
        versions = ['v1', 'v2', 'v3', 'v4']
        version_names = ['V1: Basic', 'V2: Hybrid', 'V3: Agentic', 'V4: CRAG']
        
        metrics = ['context_relevance', 'bleu', 'rouge', 
                   'hallucination_rate', 'overall_score']
        
        data = {metric: [] for metric in metrics}
        latencies = []
        
        for v in versions:
            if v in result:
                for metric in metrics:
                    data[metric].append(result[v].get(metric, 0))
                latencies.append(result[v].get('latency', 0))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Metrics comparison
        x = np.arange(len(metrics))
        width = 0.2
        
        display_metrics = ['Context_Relevance', 'BLEU', 'ROUGE', 
                          'Hallucination', 'Overall']
        
        for i, (v, name) in enumerate(zip(versions, version_names)):
            values = []
            for j, m in enumerate(metrics):
                val = data[m][i] if i < len(data[m]) else 0
                values.append(val)
            ax1.bar(x + i * width, values, width, label=name)
        
        ax1.set_xlabel('Metrics', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        query_display = query[:50] + '...' if len(query) > 50 else query
        # Replace Hindi characters with transliteration for display
        try:
            query_display.encode('ascii')
            title_query = query_display
        except UnicodeEncodeError:
            title_query = "[Hindi Query]"
            
        ax1.set_title(f'RAG Metrics Comparison\nQuery: {title_query}', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(display_metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        ax1.text(0.02, 0.98, 'Note: Lower Hallucination = Better', 
                transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Plot 2: Latency comparison
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax2.bar(version_names, latencies, color=colors)
        ax2.set_xlabel('Version', fontsize=11)
        ax2.set_ylabel('Latency (seconds)', fontsize=11)
        ax2.set_title('Response Latency Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{self.charts_dir}/query_{len(self.history)}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to {filename}")
        plt.close()
    
    def plot_aggregate_stats(self):
        """Create aggregate statistics"""
        if len(self.history) < 2:
            return
        
        versions = ['v1', 'v2', 'v3', 'v4']
        version_names = ['V1: Basic', 'V2: Hybrid', 'V3: Agentic', 'V4: CRAG']
        
        metrics = ['context_relevance', 'bleu', 'rouge', 
                   'hallucination_rate', 'overall_score']
        
        # Aggregate data
        agg_data = {v: {m: [] for m in metrics + ['latency']} for v in versions}
        
        for result in self.history:
            for v in versions:
                if v in result:
                    for m in metrics:
                        agg_data[v][m].append(result[v].get(m, 0))
                    agg_data[v]['latency'].append(result[v].get('latency', 0))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Average Overall Score
        avg_overall = [np.mean(agg_data[v]['overall_score']) for v in versions]
        bars = axes[0, 0].bar(version_names, avg_overall, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        axes[0, 0].set_ylabel('Average Overall Score', fontsize=11)
        axes[0, 0].set_title(f'Overall Performance ({len(self.history)} queries)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Metrics heatmap
        display_metrics = ['Context_Relevance', 'BLEU', 'ROUGE', 'Hallucination', 'Overall']
        heatmap_data = []
        for v in versions:
            row = []
            for m in metrics:
                val = np.mean(agg_data[v][m])
                row.append(val)
            heatmap_data.append(row)
        
        im = axes[0, 1].imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 1].set_xticks(np.arange(len(display_metrics)))
        axes[0, 1].set_yticks(np.arange(len(versions)))
        axes[0, 1].set_xticklabels(display_metrics, rotation=45, ha='right')
        axes[0, 1].set_yticklabels(version_names)
        axes[0, 1].set_title('Metrics Heatmap', fontsize=12, fontweight='bold')
        
        for i in range(len(versions)):
            for j in range(len(display_metrics)):
                axes[0, 1].text(j, i, f'{heatmap_data[i][j]:.2f}',
                               ha='center', va='center', color='black', fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: Average latency
        avg_latency = [np.mean(agg_data[v]['latency']) for v in versions]
        bars = axes[1, 0].bar(version_names, avg_latency, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        axes[1, 0].set_ylabel('Average Latency (s)', fontsize=11)
        axes[1, 0].set_title('Response Time Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Score distribution
        for i, (v, name) in enumerate(zip(versions, version_names)):
            scores = agg_data[v]['overall_score']
            axes[1, 1].hist(scores, alpha=0.6, label=name, bins=10)
        
        axes[1, 1].set_xlabel('Overall Score', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Score Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = f'{self.charts_dir}/aggregate_stats.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Aggregate stats saved to {filename}")
        plt.close()


    def plot_context_analysis(self):
        """Create context-based performance analysis"""
        if len(self.history) < 2:
            return
        
        # Group results by context
        context_data = {}
        
        for result in self.history:
            ctx = result.get('context', 'unknown')
            if ctx not in context_data:
                context_data[ctx] = {'v1': [], 'v2': [], 'v3': [], 'v4': []}
            
            for version in ['v1', 'v2', 'v3', 'v4']:
                if version in result:
                    context_data[ctx][version].append(result[version].get('overall_score', 0))
        
        # Calculate average scores per context
        contexts = list(context_data.keys())
        versions = ['v1', 'v2', 'v3', 'v4']
        version_names = ['V1: Basic', 'V2: Hybrid', 'V3: Agentic', 'V4: CRAG']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Grouped bar chart
        x = np.arange(len(contexts))
        width = 0.2
        
        for i, (v, name) in enumerate(zip(versions, version_names)):
            scores = [np.mean(context_data[ctx][v]) if context_data[ctx][v] else 0 
                    for ctx in contexts]
            ax1.bar(x + i * width, scores, width, label=name, 
                    color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][i])
        
        ax1.set_xlabel('Context Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Overall Score', fontsize=12, fontweight='bold')
        ax1.set_title('RAG Performance by Context Type', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(contexts, rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Heatmap showing best version for each context
        heatmap_data = []
        for ctx in contexts:
            row = []
            for v in versions:
                scores = context_data[ctx][v]
                avg = np.mean(scores) if scores else 0
                row.append(avg)
            heatmap_data.append(row)
        
        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(np.arange(len(version_names)))
        ax2.set_yticks(np.arange(len(contexts)))
        ax2.set_xticklabels(version_names, rotation=45, ha='right')
        ax2.set_yticklabels(contexts)
        ax2.set_title('Performance Heatmap: Context vs RAG Version', fontsize=14, fontweight='bold')
        
        # Annotate cells with scores and mark best
        for i in range(len(contexts)):
            row_max = max(heatmap_data[i])
            for j in range(len(versions)):
                score = heatmap_data[i][j]
                text_color = 'white' if score < 0.5 else 'black'
                
                # Mark best version with ⭐
                if score == row_max and score > 0:
                    text = f'{score:.3f}\n⭐'
                    fontweight = 'bold'
                else:
                    text = f'{score:.3f}'
                    fontweight = 'normal'
                    
                ax2.text(j, i, text, ha='center', va='center', 
                        color=text_color, fontsize=10, fontweight=fontweight)
        
        plt.colorbar(im, ax=ax2, label='Overall Score')
        
        # Add summary table
        summary_text = "Best RAG Version per Context:\n" + "-"*40 + "\n"
        for i, ctx in enumerate(contexts):
            best_idx = np.argmax(heatmap_data[i])
            best_version = version_names[best_idx]
            best_score = heatmap_data[i][best_idx]
            summary_text += f"{ctx}: {best_version} ({best_score:.3f})\n"
        
        fig.text(0.5, -0.05, summary_text, ha='center', fontsize=10, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filename = f'{self.charts_dir}/context_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Context analysis saved to {filename}")
        plt.close()

# ============================================================================
# MAIN SYSTEM WITH DATASET SUPPORT
# ============================================================================

class MultiVersionRAG:
    """Main system with embedding caching and dataset support"""
    
    def __init__(self, json_file: str = "products_100.json", use_gemini: bool = True):
        print("="*80)
        print("OPTIMIZED MULTI-VERSION RAG SYSTEM - WITH GROUND TRUTH EVALUATION")
        print("="*80)
        print("\n🚀 Key Features:")
        print("  • Embeddings cached to disk (generate once, reuse forever)")
        print("  • Ground truth evaluation with BLEU/ROUGE")
        print("  • Automatic dataset processing")
        print("  • Rate limiting for API calls")
        print("  • Comprehensive metrics & visualizations")
        print("="*80)
        print("\nInitializing system...")
        
        self.cache = EmbeddingCache()
        
        print("  [1/5] Loading product data...")
        loader = JSONLoader(
            file_path=json_file,
            jq_schema=".[]",
            text_content=False
        )
        self.docs = loader.load()
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.products_data = json.load(f)
        print(f"      ✅ Loaded {len(self.docs)} products")
        
        print("  [2/5] Initializing embeddings...")
        print("      ⏳ Using HuggingFace embeddings (free, no rate limits)...")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            repo_id="BAAI/bge-small-en-v1.5",
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        embedding_model_name = "BAAI/bge-small-en-v1.5"
        print("      ✅ Embeddings initialized")
        
        print("  [3/5] Loading/Creating vector store...")
        self.vector_store = self.cache.load_vector_store(json_file, embedding_model_name)
        
        if self.vector_store is None:
            print("      ⏳ Creating embeddings (first time only)...")
            self.vector_store = FAISS.from_documents(documents=self.docs, embedding=self.embeddings)
            self.cache.save_vector_store(self.vector_store, json_file, embedding_model_name)
        
        print("  [4/5] Initializing LLM...")
        if use_gemini:
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.3,
                max_output_tokens=256
            )
        else:
            llm = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-7B-Instruct",
                task="text-generation",
                max_new_tokens=256,
                temperature=0.3,
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            )
            self.model = ChatHuggingFace(llm=llm)
        print("      ✅ LLM ready")
        
        print("  [5/5] Loading/Creating BM25 index...")
        bm25_data = self.cache.load_bm25(json_file)
        
        if bm25_data is None:
            print("      ⏳ Creating BM25 index (first time only)...")
            corpus = [json.loads(d.page_content) for d in self.docs]
            tokenized_corpus = [text.lower().split() for text in [json.dumps(d) for d in corpus]]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_data = (bm25, corpus, tokenized_corpus)
            self.cache.save_bm25(bm25, corpus, tokenized_corpus, json_file)
        
        self.v1 = BasicRAG(self.vector_store, self.model)
        print("      ✅ Version 1: Basic RAG")
        
        self.v2 = HybridRAG(self.vector_store, bm25_data, self.model)
        print("      ✅ Version 2: Hybrid + Reranking")
        
        self.v3 = AgenticRAG(self.vector_store, self.products_data, self.model)
        print("      ✅ Version 3: Agentic RAG")
        
        self.v4 = CorrectiveRAG(self.vector_store, self.products_data, self.model)
        print("      ✅ Version 4: Corrective RAG")

        self.relevance_checker = QueryRelevanceChecker(self.model)
        print("      ✅ Query Relevance Checker")
        
        self.evaluator = RAGEvaluator(self.model)
        self.results_manager = ResultsManager()
        
        print("\n" + "="*80)
        print("✅ SYSTEM READY!")
        print("="*80 + "\n")
    
    def process_query(self, query: str, ground_truth: str = None, context: str = None):
        """Process query through all versions and compare"""
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        if ground_truth:
            print(f"GROUND TRUTH: {ground_truth}")
        print("="*80 + "\n")
        
        # ========================================================================
        # STEP 1: CHECK IF QUERY IS RETAIL-RELATED
        # ========================================================================
        print("🔍 Checking query relevance...")
        is_relevant, explanation = self.relevance_checker.is_retail_related(query)
        
        if not is_relevant:
            print(f"❌ {explanation}")
            print("\n" + "="*80)
            print("⚠️  QUERY REJECTED - NOT RETAIL-RELATED")
            print("="*80)
            print("\n💡 I don't have context for this query.")
            print("   Please ask me retail-related questions about:")
            print("   • Product locations and availability")
            print("   • Product recommendations")
            print("   • Prices and inventory")
            print("   • Store information")
            print("="*80 + "\n")
            return None
        
        print(f"✅ {explanation}")
        print()
        
        # ========================================================================
        # STEP 2: PROCESS QUERY THROUGH ALL RAG VERSIONS
        # ========================================================================
        results = {}
        
        # Version 1
        print("▶ VERSION 1: BASIC RAG")
        print("-" * 80)
        v1_result = self.v1.query(query)
        print(f"Response: {v1_result['response']}")
        print(f"Latency: {v1_result['latency']:.2f}s | Docs: {v1_result['num_docs']}")
        
        v1_metrics = self.evaluator.evaluate(query, v1_result['response'], 
                                            v1_result['context'], 
                                            v1_result.get('retrieved_docs', []),
                                            ground_truth)
        print(f"Metrics: Context={v1_metrics['context_relevance']:.3f}, "
            f"BLEU={v1_metrics['bleu']:.3f}, "
            f"ROUGE={v1_metrics['rouge']:.3f}, "
            f"Hallucination={v1_metrics['hallucination_rate']:.3f}, "
            f"Overall={v1_metrics['overall_score']:.3f}")
        
        results['v1'] = {**v1_result, **v1_metrics}
        
        # Version 2
        print("\n▶ VERSION 2: HYBRID RETRIEVAL + RERANKING")
        print("-" * 80)
        v2_result = self.v2.query(query)
        print(f"Response: {v2_result['response']}")
        print(f"Latency: {v2_result['latency']:.2f}s | Docs: {v2_result['num_docs']}")
        
        v2_metrics = self.evaluator.evaluate(query, v2_result['response'], 
                                            v2_result['context'],
                                            v2_result.get('retrieved_docs', []),
                                            ground_truth)
        print(f"Metrics: Context={v2_metrics['context_relevance']:.3f}, "
            f"BLEU={v2_metrics['bleu']:.3f}, "
            f"ROUGE={v2_metrics['rouge']:.3f}, "
            f"Hallucination={v2_metrics['hallucination_rate']:.3f}, "
            f"Overall={v2_metrics['overall_score']:.3f}")
        
        results['v2'] = {**v2_result, **v2_metrics}
        
        # Version 3
        print("\n▶ VERSION 3: AGENTIC RAG")
        print("-" * 80)
        v3_result = self.v3.query(query)
        print(f"Response: {v3_result['response']}")
        print(f"Latency: {v3_result['latency']:.2f}s | Tool Calls: {v3_result.get('num_tool_calls', 0)}")
        
        v3_metrics = self.evaluator.evaluate(query, v3_result['response'], 
                                            v3_result['context'],
                                            v3_result.get('retrieved_docs', []),
                                            ground_truth)
        print(f"Metrics: Context={v3_metrics['context_relevance']:.3f}, "
            f"BLEU={v3_metrics['bleu']:.3f}, "
            f"ROUGE={v3_metrics['rouge']:.3f}, "
            f"Hallucination={v3_metrics['hallucination_rate']:.3f}, "
            f"Overall={v3_metrics['overall_score']:.3f}")
        
        results['v3'] = {**v3_result, **v3_metrics}
        
        # Version 4
        print("\n▶ VERSION 4: CORRECTIVE RAG (CRAG)")
        print("-" * 80)
        v4_result = self.v4.query(query)
        print(f"Response: {v4_result['response']}")
        print(f"Latency: {v4_result['latency']:.2f}s | Relevance: {v4_result.get('avg_relevance', 0):.3f}")
        
        v4_metrics = self.evaluator.evaluate(query, v4_result['response'], 
                                            v4_result['context'],
                                            v4_result.get('retrieved_docs', []),
                                            ground_truth)
        print(f"Metrics: Context={v4_metrics['context_relevance']:.3f}, "
            f"BLEU={v4_metrics['bleu']:.3f}, "
            f"ROUGE={v4_metrics['rouge']:.3f}, "
            f"Hallucination={v4_metrics['hallucination_rate']:.3f}, "
            f"Overall={v4_metrics['overall_score']:.3f}")
        
        results['v4'] = {**v4_result, **v4_metrics}
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY COMPARISON")
        print("="*80)
        
        summary_data = []
        for v in ['v1', 'v2', 'v3', 'v4']:
            summary_data.append({
                'Version': v.upper(),
                'BLEU': f"{results[v]['bleu']:.3f}",
                'ROUGE': f"{results[v]['rouge']:.3f}",
                'Overall': f"{results[v]['overall_score']:.3f}",
                'Latency': f"{results[v]['latency']:.2f}s"
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        self.results_manager.add_result(query, results, ground_truth, context)
        
        
        print("Generating visualizations...")
        self.results_manager.plot_comparison()
        
        if len(self.results_manager.history) > 1:
            self.results_manager.plot_aggregate_stats()
        
        best_version = max(results.items(), key=lambda x: x[1]['overall_score'])
        print(f"\n🏆 BEST PERFORMING VERSION: {best_version[0].upper()} "
            f"(Overall Score: {best_version[1]['overall_score']:.3f})")
        print("="*80 + "\n")
        
        return results
    
    def run_dataset(self, dataset_file: str = "dataset.json", delay: int = 15):
        """Run evaluation on entire dataset with rate limiting"""
        print("\n" + "="*80)
        print("📊 RUNNING DATASET EVALUATION")
        print("="*80 + "\n")
        
        # Load dataset
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"✅ Loaded {len(dataset)} queries from {dataset_file}\n")
        except FileNotFoundError:
            print(f"❌ Error: {dataset_file} not found!")
            return
        
        # Process each query
        for i, item in enumerate(dataset, 1):
            query = item['query']
            ground_truth = item.get('ground_truth', None)
            context = item.get('context', 'unknown')
            
            print(f"\n{'='*80}")
            print(f"PROCESSING QUERY {i}/{len(dataset)}")
            print(f"{'='*80}")
            
            self.process_query(query, ground_truth, context)
            
            # Rate limiting - wait between queries (except for last one)
            if i < len(dataset):
                print(f"\n⏳ Waiting {delay} seconds to respect API rate limits...")
                time.sleep(delay)
        
        print("\n" + "="*80)
        print("✅ DATASET EVALUATION COMPLETE!")
        print("="*80)
        print(f"\n📊 Results saved to: {self.results_manager.results_file}")
        print(f"📈 Charts saved to: {self.results_manager.charts_dir}/")
        print("\nGenerating final aggregate statistics...")
        self.results_manager.plot_aggregate_stats()
        self.results_manager.plot_context_analysis() 
        
        print("\n✅ All done!")
    
    def run_interactive(self):
        """Interactive query mode"""
        print("\nðŸ'¬ INTERACTIVE MODE")
        print("Enter your queries below. Type 'quit' to exit, 'stats' for aggregate stats.\n")
        
        while True:
            query = input("Your Query: ").strip()
            
            if query.lower() == 'quit':
                print("\nðŸ'‹ Goodbye!")
                # Generate final charts before exiting
                if len(self.results_manager.history) > 1:
                    print("\nGenerating final statistics...")
                    self.results_manager.plot_aggregate_stats()
                    self.results_manager.plot_context_analysis()  # <-- ADD THIS LINE
                    print("âœ… Charts saved!")
                break
            elif query.lower() == 'stats':
                if len(self.results_manager.history) > 1:
                    self.results_manager.plot_aggregate_stats()
                    self.results_manager.plot_context_analysis()  # <-- ADD THIS LINE TOO
                else:
                    print("Need at least 2 queries to show stats!")
                continue
            elif not query:
                continue
            
            self.process_query(query)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    
    use_gemini = bool(os.getenv("GOOGLE_API_KEY"))
    
    if not use_gemini and not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("⚠️  ERROR: Please set GOOGLE_API_KEY or HUGGINGFACEHUB_API_TOKEN in .env file")
        return
    
    # Initialize system
    system = MultiVersionRAG("products_100.json", use_gemini=use_gemini)
    
    # Check if dataset exists
    if os.path.exists("dataset50.json"):
        print("\n📋 Found dataset.json!")
        choice = input("Run dataset evaluation? (y/n): ").strip().lower()
        
        if choice == 'y':
            delay = input("Enter delay between queries in seconds (default 15): ").strip()
            delay = int(delay) if delay.isdigit() else 15
            system.run_dataset("dataset50.json", delay=delay)
        else:
            system.run_interactive()
    else:
        print("\n⚠️  dataset.json not found. Starting interactive mode...")
        system.run_interactive()

if __name__ == "__main__":
    main()