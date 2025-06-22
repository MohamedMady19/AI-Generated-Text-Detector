"""
Custom Persistent Homology Dimension (PHD) Implementation
Uses the exact PHD implementation as specified from GPTID project
"""

import numpy as np
from scipy.spatial.distance import cdist
from threading import Thread
import logging
from typing import Dict, Optional, Union
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

logger = logging.getLogger(__name__)

MINIMAL_CLOUD = 80

def prim_tree(adj_matrix, alpha=1.0):
    """
    EXACT implementation from the provided PHD code - DO NOT MODIFY
    
    Compute Prim's tree with alpha parameter
    """
    infty = np.max(adj_matrix) + 10
    
    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty
        
        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)
        
    return s.item()

def process_string(sss):
    """
    EXACT implementation from the provided PHD code - DO NOT MODIFY
    """
    return sss.replace('\n', ' ').replace('  ', ' ')

class PHD():
    """
    EXACT implementation from the provided PHD code - DO NOT MODIFY
    
    Persistent Homology Dimension computer
    """
    def __init__(self, alpha=1.0, metric='euclidean', n_reruns=3, n_points=7, n_points_min=3):
        '''
        Initializes the instance of PH-dim computer
        Parameters:
        1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
        2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
        3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
        4) n_points --- Number of subsamples to be drawn at each subsample
        5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        '''
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False
        self.distance_matrix = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        if self.distance_matrix:
            return W[random_indices][:, random_indices]
        else:
            return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points
               
            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                if self.distance_matrix:
                    reruns[i] = prim_tree(tmp, self.alpha)
                else:
                    reruns[i] = prim_tree(cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)   
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
        
    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40, dist=False):
        '''
        Computing the PH-dim 
        Parameters:
        1) X --- point cloud of shape (n_points, n_features), or precomputed distance matrix (n_points, n_points) if parameter dist set to 'True'
        2) y --- fictional parameter to fit with Sklearn interface
        3) min_points --- size of minimal subsample to be drawn
        4) max_points --- size of maximal subsample to be drawn
        5) point_jump --- step between subsamples
        6) dist --- bool value whether X is a precomputed distance matrix
        '''
        self.distance_matrix = dist
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            threads.append(Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i]))
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        m = np.mean(ms)
        return 1 / (1 - m)


class TextToPHDConverter:
    """
    Converter to transform text into point clouds for PHD computation
    Following GPTID methodology
    """
    
    def __init__(self, method='tfidf', max_features=1000, min_sentences=5):
        """
        Initialize text to PHD converter
        
        Args:
            method: Method for text representation ('tfidf', 'sentence_embeddings')
            max_features: Maximum features for TF-IDF
            min_sentences: Minimum sentences required for PHD computation
        """
        self.method = method
        self.max_features = max_features
        self.min_sentences = min_sentences
        self.vectorizer = None
        self.nlp = None
        
        if method == 'sentence_embeddings':
            try:
                # Try to load spaCy model for sentence embeddings
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model not found, falling back to TF-IDF")
                self.method = 'tfidf'
    
    def text_to_point_cloud(self, text: str) -> Optional[np.ndarray]:
        """
        Convert text to point cloud representation
        
        Args:
            text: Input text
            
        Returns:
            Point cloud as numpy array or None if conversion fails
        """
        try:
            if self.method == 'tfidf':
                return self._tfidf_to_point_cloud(text)
            elif self.method == 'sentence_embeddings':
                return self._sentences_to_point_cloud(text)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        except Exception as e:
            logger.warning(f"Failed to convert text to point cloud: {e}")
            return None
    
    def _tfidf_to_point_cloud(self, text: str) -> Optional[np.ndarray]:
        """Convert text to point cloud using TF-IDF sentence vectors"""
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) < self.min_sentences:
            logger.debug(f"Not enough sentences for PHD: {len(sentences)} < {self.min_sentences}")
            return None
        
        # Create TF-IDF vectors for sentences
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            point_cloud = tfidf_matrix.toarray()
            
            # Ensure we have enough points and features
            if point_cloud.shape[0] < MINIMAL_CLOUD:
                logger.debug(f"Point cloud too small: {point_cloud.shape[0]} < {MINIMAL_CLOUD}")
                return None
            
            return point_cloud
            
        except Exception as e:
            logger.warning(f"TF-IDF conversion failed: {e}")
            return None
    
    def _sentences_to_point_cloud(self, text: str) -> Optional[np.ndarray]:
        """Convert text to point cloud using sentence embeddings"""
        if self.nlp is None:
            return self._tfidf_to_point_cloud(text)  # Fallback
        
        # Process text with spaCy
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        if len(sentences) < self.min_sentences:
            logger.debug(f"Not enough sentences for PHD: {len(sentences)} < {self.min_sentences}")
            return None
        
        # Get sentence embeddings
        sentence_vectors = []
        for sentence in sentences:
            sent_doc = self.nlp(sentence)
            if sent_doc.vector.any():  # Check if vector is not zero
                sentence_vectors.append(sent_doc.vector)
        
        if len(sentence_vectors) < self.min_sentences:
            logger.debug("Not enough valid sentence vectors")
            return None
        
        point_cloud = np.array(sentence_vectors)
        
        if point_cloud.shape[0] < MINIMAL_CLOUD:
            logger.debug(f"Point cloud too small: {point_cloud.shape[0]} < {MINIMAL_CLOUD}")
            return None
        
        return point_cloud
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy)
        import re
        
        # Clean and process text first
        text = process_string(text)
        
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences


def extract_phd_features(text: str, config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract PHD features from text using the custom PHD implementation
    
    Args:
        text: Input text
        config: Configuration dictionary with PHD parameters
        
    Returns:
        Dictionary of PHD features
    """
    if config is None:
        config = {}
    
    # Default PHD parameters from config
    alpha = config.get('PHD_ALPHA', 1.0)
    metric = config.get('PHD_METRIC', 'euclidean')
    n_reruns = config.get('PHD_N_RERUNS', 3)
    n_points = config.get('PHD_N_POINTS', 7)
    n_points_min = config.get('PHD_N_POINTS_MIN', 3)
    min_points = config.get('PHD_MIN_POINTS', 50)
    max_points = config.get('PHD_MAX_POINTS', 512)
    point_jump = config.get('PHD_POINT_JUMP', 40)
    
    features = {
        'ph_dimension': 0.0,
        'ph_dimension_tfidf': 0.0,
        'ph_dimension_embeddings': 0.0,
        'ph_valid': 0.0,
        'ph_point_cloud_size': 0.0,
        'ph_computation_success': 0.0
    }
    
    try:
        # Initialize PHD computer
        phd_computer = PHD(
            alpha=alpha,
            metric=metric,
            n_reruns=n_reruns,
            n_points=n_points,
            n_points_min=n_points_min
        )
        
        # Try TF-IDF method
        converter_tfidf = TextToPHDConverter(method='tfidf')
        point_cloud_tfidf = converter_tfidf.text_to_point_cloud(text)
        
        if point_cloud_tfidf is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ph_dim_tfidf = phd_computer.fit_transform(
                        point_cloud_tfidf,
                        min_points=min(min_points, point_cloud_tfidf.shape[0] // 2),
                        max_points=min(max_points, point_cloud_tfidf.shape[0]),
                        point_jump=point_jump
                    )
                    features['ph_dimension_tfidf'] = float(ph_dim_tfidf)
                    features['ph_dimension'] = float(ph_dim_tfidf)  # Primary PHD value
                    features['ph_point_cloud_size'] = float(point_cloud_tfidf.shape[0])
                    features['ph_computation_success'] = 1.0
                    features['ph_valid'] = 1.0
                    
                    logger.debug(f"PHD TF-IDF computation successful: {ph_dim_tfidf}")
                    
            except Exception as e:
                logger.warning(f"PHD TF-IDF computation failed: {e}")
        
        # Try sentence embeddings method if available
        converter_embeddings = TextToPHDConverter(method='sentence_embeddings')
        point_cloud_embeddings = converter_embeddings.text_to_point_cloud(text)
        
        if point_cloud_embeddings is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ph_dim_embeddings = phd_computer.fit_transform(
                        point_cloud_embeddings,
                        min_points=min(min_points, point_cloud_embeddings.shape[0] // 2),
                        max_points=min(max_points, point_cloud_embeddings.shape[0]),
                        point_jump=point_jump
                    )
                    features['ph_dimension_embeddings'] = float(ph_dim_embeddings)
                    
                    # Use embeddings as primary if TF-IDF failed
                    if features['ph_computation_success'] == 0.0:
                        features['ph_dimension'] = float(ph_dim_embeddings)
                        features['ph_point_cloud_size'] = float(point_cloud_embeddings.shape[0])
                        features['ph_computation_success'] = 1.0
                        features['ph_valid'] = 1.0
                    
                    logger.debug(f"PHD embeddings computation successful: {ph_dim_embeddings}")
                    
            except Exception as e:
                logger.warning(f"PHD embeddings computation failed: {e}")
        
        # If no method succeeded, log it
        if features['ph_computation_success'] == 0.0:
            logger.debug("All PHD computation methods failed")
    
    except Exception as e:
        logger.warning(f"PHD feature extraction failed: {e}")
    
    return features


def safe_phd_extractor(feature_name: str, default_features: Dict[str, float]):
    """
    Decorator for safe PHD feature extraction with error handling
    """
    def decorator(func):
        def wrapper(text: str, doc=None, config: Optional[Dict] = None) -> Dict[str, float]:
            try:
                return func(text, doc, config)
            except Exception as e:
                logger.warning(f"PHD feature extraction '{feature_name}' failed: {e}")
                return default_features.copy()
        return wrapper
    return decorator


@safe_phd_extractor('custom_phd_features', {
    'ph_dimension': 0.0,
    'ph_dimension_tfidf': 0.0,
    'ph_dimension_embeddings': 0.0,
    'ph_valid': 0.0,
    'ph_point_cloud_size': 0.0,
    'ph_computation_success': 0.0
})
def custom_phd_features(text: str, doc=None, config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Main function to extract custom PHD features
    
    Args:
        text: Input text
        doc: spaCy document (optional, for compatibility)
        config: Configuration dictionary
        
    Returns:
        Dictionary of PHD features
    """
    return extract_phd_features(text, config)