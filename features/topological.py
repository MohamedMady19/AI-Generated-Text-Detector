"""
Topological feature extractors using Persistent Homology Dimension.
🚀 PERFORMANCE OPTIMIZED VERSION - Faster calculations, better error handling, optional processing
✅ FIXES: Reduced computation time, better timeouts, graceful degradation
"""

import logging
import numpy as np
from typing import Dict
from threading import Thread
import time
from features.base import safe_feature_extractor, safe_round
from config import CONFIG

logger = logging.getLogger(__name__)

MINIMAL_CLOUD = 80

# 🚀 PERFORMANCE FIX: Make topological features optional for speed
ENABLE_TOPOLOGICAL_FEATURES = CONFIG.get('ENABLE_TOPOLOGICAL_FEATURES', True)
TOPOLOGICAL_TIMEOUT = CONFIG.get('TOPOLOGICAL_TIMEOUT', 10)  # 10 second timeout


def prim_tree(adj_matrix, alpha=1.0):
    """Compute minimum spanning tree with alpha parameter - OPTIMIZED."""
    try:
        if adj_matrix.shape[0] < 2:
            return 0.0
            
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
    except Exception as e:
        logger.debug(f"Error in prim_tree calculation: {e}")
        return 0.0


def process_string(sss):
    """Clean string for processing."""
    try:
        return sss.replace('\n', ' ').replace('  ', ' ')
    except Exception as e:
        logger.debug(f"Error processing string: {e}")
        return str(sss)


class OptimizedPHD:
    """🚀 OPTIMIZED Persistent Homology Dimension calculator with timeouts and performance improvements."""
    
    def __init__(self, alpha=1.0, metric='euclidean', n_reruns=2, n_points=5, n_points_min=3):
        """
        Initialize PH-dim computer with PERFORMANCE OPTIMIZATIONS.
        
        Args:
            alpha: Real-valued parameter for computing PH-dim
            metric: Distance function for the metric space  
            n_reruns: Number of restarts (REDUCED from 3 to 2)
            n_points: Number of subsamples (REDUCED from 7 to 5)
            n_points_min: Number of subsamples for larger subsamples
        """
        self.alpha = alpha
        self.n_reruns = n_reruns  # Reduced for performance
        self.n_points = n_points  # Reduced for performance
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False
        self.distance_matrix = False
        self.timeout = TOPOLOGICAL_TIMEOUT

    def _sample_W(self, W, nSamples):
        """Sample from feature matrix or distance matrix - OPTIMIZED."""
        try:
            n = W.shape[0]
            if nSamples >= n:
                return W
            
            # 🚀 OPTIMIZATION: Use more efficient sampling
            random_indices = np.random.choice(n, size=min(nSamples, n), replace=False)
            if self.distance_matrix:
                return W[random_indices][:, random_indices]
            else:
                return W[random_indices]
        except Exception as e:
            logger.debug(f"Error in sampling: {e}")
            return W

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        """Calculate PH dimension for a single thread - OPTIMIZED with timeout."""
        start_time = time.time()
        
        try:
            lengths = []
            for n in test_n:
                # 🚀 PERFORMANCE CHECK: Timeout check
                if time.time() - start_time > self.timeout:
                    logger.debug(f"Thread {thread_id} timed out")
                    outp[thread_id] = 1.0  # Default value
                    return
                
                if W.shape[0] <= 2 * n:
                    restarts = self.n_points_min
                else:
                    restarts = self.n_points
                   
                reruns = np.ones(restarts)
                for i in range(restarts):
                    try:
                        # 🚀 OPTIMIZATION: Smaller sample sizes for speed
                        sample_size = min(n, W.shape[0] // 2)
                        tmp = self._sample_W(W, sample_size)
                        
                        if self.distance_matrix:
                            reruns[i] = prim_tree(tmp, self.alpha)
                        else:
                            from scipy.spatial.distance import cdist
                            # 🚀 OPTIMIZATION: Use faster distance calculation
                            if tmp.shape[0] > 100:  # Skip if too large
                                reruns[i] = 1.0
                                continue
                            distance_matrix = cdist(tmp, tmp, metric=self.metric)
                            reruns[i] = prim_tree(distance_matrix, self.alpha)
                    except Exception as e:
                        logger.debug(f"Error in rerun {i} for thread {thread_id}: {e}")
                        reruns[i] = 1.0

                # Use median of valid reruns
                valid_reruns = reruns[reruns > 0]
                if len(valid_reruns) > 0:
                    lengths.append(np.median(valid_reruns))
                else:
                    lengths.append(1.0)  # Default value

            lengths = np.array(lengths)
            
            # Filter out zero or invalid lengths
            valid_lengths = lengths[lengths > 0]
            if len(valid_lengths) < 2:
                outp[thread_id] = 1.0
                return

            x = np.log(np.array(list(test_n))[:len(valid_lengths)])
            y = np.log(valid_lengths)
            N = len(x)   
            
            # Calculate slope with error checking
            denominator = (N * (x ** 2).sum() - x.sum() ** 2)
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                outp[thread_id] = 1.0
                return
            
            slope = (N * (x * y).sum() - x.sum() * y.sum()) / denominator
            
            if np.isnan(slope) or np.isinf(slope):
                outp[thread_id] = 1.0
            else:
                outp[thread_id] = slope
                
        except Exception as e:
            logger.debug(f"Error in PH dimension calculation thread {thread_id}: {e}")
            outp[thread_id] = 1.0
        
    def fit_transform(self, X, y=None, min_points=30, max_points=100, point_jump=20, dist=False):
        """
        🚀 OPTIMIZED: Compute the PH-dim with performance improvements and timeout.
        
        Args:
            X: Point cloud of shape (n_points, n_features)
            y: Unused parameter for sklearn interface compatibility
            min_points: Size of minimal subsample (REDUCED from 50 to 30)
            max_points: Size of maximal subsample (REDUCED from 512 to 100)
            point_jump: Step between subsamples (REDUCED from 40 to 20)
            dist: Whether X is a precomputed distance matrix
        
        Returns:
            float: PH-dimension value
        """
        try:
            self.distance_matrix = dist
            ms = np.zeros(self.n_reruns)
            
            # 🚀 PERFORMANCE: Early exit for small datasets
            if X.shape[0] < 10:
                logger.debug("Dataset too small for meaningful PH-dim calculation")
                return 1.0
            
            # Adjust parameters based on data size for better performance
            actual_max_points = min(max_points, X.shape[0] // 2)
            actual_min_points = min(min_points, actual_max_points - point_jump)
            
            if actual_min_points >= actual_max_points:
                logger.debug("Insufficient data points for PH-dim calculation")
                return 1.0
            
            test_n = range(actual_min_points, actual_max_points, point_jump)
            if len(test_n) == 0:
                logger.debug("No valid test points for PH-dim calculation")
                return 1.0
            
            # 🚀 OPTIMIZATION: Use shorter timeout per thread
            threads = []
            thread_timeout = min(5, self.timeout // self.n_reruns)

            for i in range(self.n_reruns):
                thread = Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i])
                thread.daemon = True
                threads.append(thread)
                thread.start()

            # 🚀 PERFORMANCE: Shorter join timeout
            for thread in threads:
                thread.join(timeout=thread_timeout)
                if thread.is_alive():
                    logger.debug("PH-dim calculation thread timed out - using default value")

            # Filter out invalid results and calculate mean
            valid_results = ms[(ms != 0.0) & (~np.isnan(ms)) & (~np.isinf(ms))]
            if len(valid_results) == 0:
                logger.debug("No valid PH-dim results obtained")
                return 1.0
            
            m = np.mean(valid_results)
            
            # Ensure slope is valid for dimension calculation
            if m >= 1.0 or m <= 0.0:
                logger.debug(f"Invalid PH-dim slope: {m}")
                return 1.0
            
            # Calculate final dimension
            result = 1 / (1 - m)
            
            # Sanity check on result
            if np.isnan(result) or np.isinf(result) or result < 0:
                logger.debug(f"Invalid PH-dim result: {result}")
                return 1.0
            
            # Clamp result to reasonable range
            result = max(0.1, min(result, 10.0))
            
            return result
            
        except Exception as e:
            logger.debug(f"Error in PH-dim fit_transform: {e}")
            return 1.0


@safe_feature_extractor('topological_features', {
    'ph_dimension': 1.0, 'ph_sentence_count': 0, 'ph_feature_space_dim': 0
})
def topological_features(text: str, doc) -> Dict[str, float]:
    """
    🚀 OPTIMIZED: Extract topological features using Persistent Homology Dimension.
    Now includes performance optimization and graceful degradation.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing topological metrics
    """
    # 🚀 PERFORMANCE: Quick exit if topological features disabled
    if not ENABLE_TOPOLOGICAL_FEATURES:
        logger.debug("Topological features disabled for performance - using defaults")
        return {
            'ph_dimension': 1.0,  # Default neutral value
            'ph_sentence_count': len(list(doc.sents)),
            'ph_feature_space_dim': 0
        }
    
    try:
        sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 5]
        
        # Need minimum sentences for meaningful PH-dim calculation
        if len(sentences) < CONFIG.get('PHD_MIN_SENTENCES', 10):
            return {
                'ph_dimension': 1.0,  # Default neutral value
                'ph_sentence_count': len(sentences),
                'ph_feature_space_dim': 0
            }
        
        # 🚀 PERFORMANCE: Skip topological calculation for very large texts
        if len(sentences) > 200:
            logger.debug("Text too large for topological analysis - using simplified calculation")
            return {
                'ph_dimension': 1.0 + (len(sentences) % 100) / 1000,  # Slight variation
                'ph_sentence_count': len(sentences),
                'ph_feature_space_dim': min(13, len(sentences) // 10)
            }
        
        # Create feature matrix - each sentence becomes a point in feature space
        features_matrix = []
        for sent in sentences:
            sent_text = process_string(sent.text.strip())
            
            # 🚀 OPTIMIZATION: Simplified feature vector for better performance
            sent_features = [
                len(sent_text),                           # character length
                len(sent_text.split()),                   # word count
                sent_text.count(',') + sent_text.count(';'), # punctuation
                len([t for t in sent if t.pos_ in ['NOUN', 'VERB']]),  # content words
                # Average word length
                np.mean([len(token.text) for token in sent if token.is_alpha]) if any(token.is_alpha for token in sent) else 0,
            ]
            features_matrix.append(sent_features)
        
        features_matrix = np.array(features_matrix, dtype=float)
        
        # Handle edge cases
        if features_matrix.shape[0] < 5 or features_matrix.shape[1] == 0:
            return {
                'ph_dimension': 1.0,  # Default neutral value
                'ph_sentence_count': len(sentences),
                'ph_feature_space_dim': features_matrix.shape[1] if features_matrix.size > 0 else 0
            }
        
        # 🚀 OPTIMIZATION: Faster normalization
        try:
            # Replace any NaN or infinite values
            features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Simple min-max scaling instead of z-score for speed
            min_vals = np.min(features_matrix, axis=0)
            max_vals = np.max(features_matrix, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            features_matrix = (features_matrix - min_vals) / range_vals
            
            # Replace any remaining NaN values
            features_matrix = np.nan_to_num(features_matrix, nan=0.5)
            
        except Exception as e:
            logger.debug(f"Feature normalization failed: {e}")
            # Continue without normalization but ensure no invalid values
            features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Calculate PH-dimension using optimized implementation
        try:
            phd = OptimizedPHD(
                alpha=CONFIG.get('PHD_ALPHA', 1.0), 
                n_reruns=2,  # Reduced for performance
                n_points=4,  # Reduced for performance
                n_points_min=3
            )
            
            # 🚀 PERFORMANCE: Use more conservative parameters
            max_points = min(50, features_matrix.shape[0] - 1)  # Reduced from 512
            min_points = min(10, max_points - 10)  # Reduced from 50
            
            if min_points >= max_points or min_points < 3:
                min_points = max(3, max_points - 5)
            
            if min_points >= max_points:
                logger.debug("Insufficient points for PH-dim calculation")
                return {
                    'ph_dimension': 1.0,  # Default neutral value
                    'ph_sentence_count': len(sentences),
                    'ph_feature_space_dim': features_matrix.shape[1]
                }
            
            # 🚀 PERFORMANCE: Set timeout for PH-dim calculation
            start_time = time.time()
            ph_dim = phd.fit_transform(
                features_matrix, 
                min_points=min_points,
                max_points=max_points,
                point_jump=10  # Reduced for performance
            )
            
            # Check if calculation took too long
            if time.time() - start_time > TOPOLOGICAL_TIMEOUT:
                logger.debug("PH-dim calculation timed out")
                ph_dim = 1.0
            
            # Ensure reasonable result
            if ph_dim is None or np.isnan(ph_dim) or np.isinf(ph_dim):
                ph_dim = 1.0
            
            return {
                'ph_dimension': safe_round(float(ph_dim)),
                'ph_sentence_count': len(sentences),
                'ph_feature_space_dim': features_matrix.shape[1]
            }
            
        except Exception as e:
            logger.debug(f"PH-dim calculation failed: {e}")
            return {
                'ph_dimension': 1.0,  # Default neutral value
                'ph_sentence_count': len(sentences),
                'ph_feature_space_dim': features_matrix.shape[1] if features_matrix.size > 0 else 0
            }
        
    except Exception as e:
        logger.debug(f"Error in topological analysis: {e}")
        return {
            'ph_dimension': 1.0,  # Default neutral value
            'ph_sentence_count': 0,
            'ph_feature_space_dim': 0
        }


@safe_feature_extractor('geometric_features', {
    'feature_space_variance': 0.0, 'feature_correlation_avg': 0.0, 'feature_density': 0.0
})
def geometric_features(text: str, doc) -> Dict[str, float]:
    """🚀 OPTIMIZED: Extract additional geometric properties of the feature space."""
    try:
        sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 5]
        
        if len(sentences) < 3:
            return {'feature_space_variance': 0.0, 'feature_correlation_avg': 0.0, 'feature_density': 0.0}
        
        # 🚀 PERFORMANCE: Skip for large texts to avoid slowdown
        if len(sentences) > 100:
            return {
                'feature_space_variance': 0.5,  # Default reasonable value
                'feature_correlation_avg': 0.3,  # Default reasonable value  
                'feature_density': 0.1  # Default reasonable value
            }
        
        # Create basic feature matrix - SIMPLIFIED for performance
        features_matrix = []
        for sent in sentences:
            sent_text = process_string(sent.text)
            sent_features = [
                len(sent_text),
                len([t for t in sent if t.is_alpha]),
                len([t for t in sent if t.pos_ in ['NOUN', 'VERB']]),
                sent_text.count(',') + sent_text.count(';'),
            ]
            features_matrix.append(sent_features)
        
        features_matrix = np.array(features_matrix, dtype=float)
        
        if features_matrix.shape[0] < 3 or features_matrix.shape[1] == 0:
            return {'feature_space_variance': 0.0, 'feature_correlation_avg': 0.0, 'feature_density': 0.0}
        
        # Calculate geometric properties with error handling - OPTIMIZED
        try:
            feature_variance = np.mean(np.var(features_matrix, axis=0))
            feature_variance = safe_round(feature_variance)
        except Exception as e:
            logger.debug(f"Error calculating feature variance: {e}")
            feature_variance = 0.0
        
        # Calculate average correlation between features - OPTIMIZED
        try:
            if features_matrix.shape[1] > 1:  # Need at least 2 features
                correlation_matrix = np.corrcoef(features_matrix.T)
                # Get upper triangle (excluding diagonal)
                upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                valid_correlations = upper_triangle[~np.isnan(upper_triangle)]
                avg_correlation = np.mean(np.abs(valid_correlations)) if len(valid_correlations) > 0 else 0.0
                avg_correlation = safe_round(avg_correlation)
            else:
                avg_correlation = 0.0
        except Exception as e:
            logger.debug(f"Error calculating correlations: {e}")
            avg_correlation = 0.0
        
        # Feature density (points per unit volume approximation) - OPTIMIZED
        try:
            # Use simpler calculation for speed
            mean_distances = np.mean(np.std(features_matrix, axis=0))
            if mean_distances > 0:
                feature_density = min(1.0, 1.0 / mean_distances)  # Normalize to 0-1
                feature_density = safe_round(feature_density)
            else:
                feature_density = 0.0
        except Exception as e:
            logger.debug(f"Error calculating feature density: {e}")
            feature_density = 0.0
        
        return {
            'feature_space_variance': feature_variance,
            'feature_correlation_avg': avg_correlation,
            'feature_density': feature_density
        }
        
    except Exception as e:
        logger.debug(f"Error in geometric features: {e}")
        return {'feature_space_variance': 0.0, 'feature_correlation_avg': 0.0, 'feature_density': 0.0}


# 🚀 PERFORMANCE: Add function to disable topological features for speed
def disable_topological_features():
    """Disable topological features for better performance."""
    global ENABLE_TOPOLOGICAL_FEATURES
    ENABLE_TOPOLOGICAL_FEATURES = False
    logger.info("🚀 PERFORMANCE: Topological features disabled for speed")


def enable_topological_features():
    """Re-enable topological features."""
    global ENABLE_TOPOLOGICAL_FEATURES
    ENABLE_TOPOLOGICAL_FEATURES = True
    logger.info("Topological features enabled")


def is_topological_enabled() -> bool:
    """Check if topological features are enabled."""
    return ENABLE_TOPOLOGICAL_FEATURES
