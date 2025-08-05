# src/models/bayesian_learner.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from .generator import Simple2LayerReLU
from .surprise_detector import SurpriseDetector

class RecursiveBayesianLearner:
    """
    Complete implementation of adaptive recursive Bayesian learning framework
    
    Key Features:
    - LC-NE inspired surprise detection with adaptive thresholds
    - Ego vehicle embodied learning for rule acquisition
    - Two-timescale dynamics (fast latent, slow generator updates)
    - Emergent bias tracking through reconstruction error patterns
    - Computer vision processing of fork intersection scenes
    - Danger avoidance task with yellow/white background detection
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize generative prior (2-layer ReLU)
        scene_size = config['visual_parameters']['scene_size']
        self.generator = Simple2LayerReLU(
            latent_dim=config['model_parameters']['latent_dim'],
            output_dim=np.prod(scene_size),
            hidden_dim=config['model_parameters']['generator_hidden']
        )
        
        # Initialize surprise detection system
        self.surprise_detector = SurpriseDetector(
            baseline_window=config['model_parameters']['reconstruction_baseline_window']
        )
        
        # Learning state tracking
        self.reconstruction_history = []
        self.context_uncertainty_history = []
        self.performance_history = []
        self.surprise_history = []
        self.action_history = []
        
        # Ego vehicle state for embodied learning
        self.ego_vehicle_state = {
            'recent_danger_encounters': [],
            'rule_confidence': 0.5,
            'conservative_bias': 0.0,
            'volatility_estimate': 0.5
        }
        
        # Rule learning system
        self.danger_avoidance_strength = 0.5
        self.rule_violation_memory = []
        
        # Adaptive learning parameters
        self.base_learning_rate = config['model_parameters']['learning_rate']
        self.adaptation_rate = config['biological_parameters']['adaptation_rate']
        
    def process_scene_sequence(self, scenes: torch.Tensor, 
                             metadata: List[Dict]) -> Dict:
        """Process continuous sequence of fork intersection scenes"""
        results = {
            'surprise_signals': [],
            'adaptive_thresholds': [],
            'reconstruction_errors': [],
            'actions': [],
            'correct_actions': [],
            'context_uncertainty': [],
            'performance': [],
            'ego_states': [],
            'rule_violations': [],
            'learning_rates': []
        }
        
        for step, (scene, meta) in enumerate(zip(scenes, metadata)):
            # Process single scene through complete framework
            step_results = self._process_single_scene(scene, meta, step)
            
            # Store all results for analysis
            for key, value in step_results.items():
                results[key].append(value)
                
        return results
    
    def _process_single_scene(self, scene: torch.Tensor, metadata: Dict, 
                            step: int) -> Dict:
        """
        Process single scene through recursive Bayesian framework with ego vehicle perspective
        """
        
        # 1. Establish ego vehicle context (enables rule understanding)
        ego_state = self._update_ego_vehicle_state(scene, metadata, step)
        
        # 2. Compute reconstruction error via generative prior
        reconstruction_error = self.generator.compute_reconstruction_error(scene)
        self.reconstruction_history.append(reconstruction_error)
        
        # 3. Compute emergent context uncertainty
        context_uncertainty = self._compute_context_uncertainty()
        prediction_confidence = self._compute_prediction_confidence()
        self.context_uncertainty_history.append(context_uncertainty)
        
        # 4. Adaptive surprise threshold (continuous function of uncertainty)
        adaptive_threshold = self.surprise_detector.adaptive_surprise_threshold(
            context_uncertainty, prediction_confidence, ego_state['volatility_regime']
        )
        
        # 5. Compute surprise signal with adaptive threshold
        surprise = self.surprise_detector.compute_surprise(
            reconstruction_error, 
            self.reconstruction_history,
            context_uncertainty=context_uncertainty,
            prediction_confidence=prediction_confidence
        )
        self.surprise_history.append(surprise)
        
        # 6. Make ego vehicle action decision with rule learning
        action = self._make_ego_vehicle_decision(scene, metadata, surprise, ego_state)
        self.action_history.append(action)
        
        # 7. Get feedback and experience consequences
        feedback = self._get_feedback(action, metadata['correct_action'])
        self.performance_history.append(feedback)
        
        # 8. Detect rule violations for learning
        rule_violation = self._detect_rule_violation(action, metadata, ego_state)
        
        # 9. Update beliefs based on embodied experience
        learning_rate = self._update_beliefs_with_ego_experience(
            surprise, feedback, reconstruction_error, ego_state, adaptive_threshold, rule_violation
        )
        
        return {
            'surprise_signal': surprise,
            'adaptive_threshold': adaptive_threshold,
            'reconstruction_error': reconstruction_error,
            'action': action,
            'correct_action': metadata['correct_action'],
            'context_uncertainty': context_uncertainty,
            'performance': feedback,
            'ego_state': ego_state.copy(),
            'rule_violation': rule_violation,
            'learning_rate': learning_rate
        }
    
    def _update_ego_vehicle_state(self, scene: torch.Tensor, metadata: Dict, step: int) -> Dict:
        """Update ego vehicle state for embodied rule learning"""
        
        # Detect current scene characteristics
        danger_detected = self._detect_danger_background(scene)
        scene_difficulty = self._assess_scene_difficulty(scene)
        
        # Update danger encounter history
        if danger_detected:
            self.ego_vehicle_state['recent_danger_encounters'].append(step)
            # Keep only recent encounters (last 100 steps)
            self.ego_vehicle_state['recent_danger_encounters'] = \
                self.ego_vehicle_state['recent_danger_encounters'][-100:]
        
        # Estimate current volatility regime
        volatility_regime = self._estimate_volatility_regime()
        
        # Update rule confidence based on recent performance
        self._update_rule_confidence()
        
        # Update conservative bias based on recent rule violations
        self._update_conservative_bias()
        
        return {
            'position': 'intersection_approach',
            'speed': 'moderate',
            'visibility': scene_difficulty,
            'volatility_regime': volatility_regime,
            'danger_frequency': len(self.ego_vehicle_state['recent_danger_encounters']) / 100,
            'rule_confidence': self.ego_vehicle_state['rule_confidence'],
            'conservative_bias': self.ego_vehicle_state['conservative_bias'],
            'recent_performance': np.mean(self.performance_history[-10:]) if len(self.performance_history) >= 10 else 0.5
        }
    
    def _compute_context_uncertainty(self) -> float:
        """Compute emergent context uncertainty from reconstruction error patterns"""
        if len(self.reconstruction_history) < 10:
            return 0.5  # Initial uncertainty
        
        # Use variance in recent reconstruction errors as uncertainty proxy
        recent_errors = self.reconstruction_history[-20:]
        recent_surprises = self.surprise_history[-20:] if len(self.surprise_history) >= 20 else [0.0] * 20
        
        # Combine reconstruction error variance with surprise volatility
        error_uncertainty = np.var(recent_errors) / (np.mean(recent_errors) + 1e-6)
        surprise_uncertainty = np.var(recent_surprises) if len(recent_surprises) > 1 else 0.0
        
        # Weight combination
        uncertainty = 0.7 * error_uncertainty + 0.3 * surprise_uncertainty
        
        return np.clip(uncertainty, 0.0, 1.0)
    
    def _compute_prediction_confidence(self) -> float:
        """Compute prediction confidence from recent performance"""
        if len(self.performance_history) < 5:
            return 0.5
        
        recent_performance = self.performance_history[-10:]
        confidence = np.mean(recent_performance)
        
        # Adjust for consistency
        consistency = 1.0 - np.var(recent_performance)
        confidence = 0.8 * confidence + 0.2 * consistency
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _estimate_volatility_regime(self) -> str:
        """Estimate current environmental volatility regime"""
        if len(self.surprise_history) < 20:
            return 'moderate'
        
        recent_surprise_rate = np.mean([abs(s) > 0.5 for s in self.surprise_history[-20:]])
        
        if recent_surprise_rate > 0.4:
            return 'volatile'
        elif recent_surprise_rate < 0.1:
            return 'stable'
        else:
            return 'moderate'
    
    def _make_ego_vehicle_decision(self, scene: torch.Tensor, metadata: Dict, 
                                 surprise: float, ego_state: Dict) -> str:
        """
        Make ego vehicle decision with learned rule understanding
        """
        
        # Detect visual features
        danger_detected = self._detect_danger_background(scene)
        left_cue_detected = self._detect_left_orientation(scene)
        
        # Apply rule hierarchy with uncertainty modulation
        if danger_detected:
            # Danger avoidance rule with confidence modulation
            base_action = 'right' if left_cue_detected else 'left'
            
            # Modulate based on rule confidence and surprise
            rule_strength = self.ego_vehicle_state['rule_confidence']
            if surprise > ego_state.get('surprise_threshold', 0.5):
                # High surprise → more conservative (stronger rule application)
                rule_strength = min(1.0, rule_strength * 1.2)
            
            # Apply conservative bias when uncertain
            if ego_state['rule_confidence'] < 0.7:
                conservative_action = self._apply_conservative_strategy(base_action, ego_state)
                action = conservative_action
            else:
                action = base_action
                
        else:
            # Safe condition: follow bar direction with bias consideration
            base_action = 'left' if left_cue_detected else 'right'
            
            # Apply learned biases from reconstruction error patterns
            action = self._apply_emergent_bias(base_action, ego_state)
        
        return action
    
    def _detect_danger_background(self, scene: torch.Tensor) -> bool:
        """Detect danger sign from yellow background vs white/gray"""
        mean_intensity = torch.mean(scene).item()
        # Yellow background has higher intensity than gray/white
        return mean_intensity > 0.6
    
    def _detect_left_orientation(self, scene: torch.Tensor) -> bool:
        """Detect left-pointing vs right-pointing oriented bars"""
        h, w = scene.shape
        center_region = scene[h//2-8:h//2+8, w//2-8:w//2+8]
        
        if center_region.numel() == 0:
            return False
        
        # Gradient-based orientation detection
        grad_x = torch.diff(center_region, dim=1)
        grad_y = torch.diff(center_region, dim=0)
        
        if grad_x.numel() == 0 or grad_y.numel() == 0:
            return False
        
        # Left-pointing bars have specific gradient correlation pattern
        try:
            # Align dimensions for correlation
            min_size = min(grad_x.shape[0], grad_y.shape[0])
            grad_x_flat = grad_x[:min_size, :].flatten()
            grad_y_flat = grad_y[:min_size, :].flatten()
            
            if len(grad_x_flat) > 1 and len(grad_y_flat) > 1:
                correlation = torch.corrcoef(torch.stack([grad_x_flat, grad_y_flat]))[0,1]
                return correlation.item() > 0
            else:
                return False
        except:
            return False
    
    def _assess_scene_difficulty(self, scene: torch.Tensor) -> float:
        """Assess visual difficulty of current scene"""
        # Use variance and edge density as difficulty proxy
        intensity_var = torch.var(scene).item()
        edges = torch.abs(torch.diff(scene, dim=0)).sum() + torch.abs(torch.diff(scene, dim=1)).sum()
        edge_density = edges.item() / scene.numel()
        
        difficulty = 1.0 - (intensity_var + edge_density) / 2.0
        return np.clip(difficulty, 0.0, 1.0)
    
    def _apply_conservative_strategy(self, base_action: str, ego_state: Dict) -> str:
        """Apply conservative bias when rule confidence is low"""
        conservative_bias = ego_state['conservative_bias']
        
        # When uncertain, prefer right turns (arbitrary but consistent choice)
        if conservative_bias > 0.3:
            return 'right'
        else:
            return base_action
    
    def _apply_emergent_bias(self, base_action: str, ego_state: Dict) -> str:
        """Apply emergent directional bias learned from reconstruction patterns"""
        
        # Extract bias from recent reconstruction error patterns
        if len(self.reconstruction_history) < 20:
            return base_action
        
        # Analyze correlation between actions and reconstruction errors
        recent_actions = self.action_history[-20:]
        recent_errors = self.reconstruction_history[-20:]
        
        if len(recent_actions) < 20:
            return base_action
        
        # Simple bias estimation
        left_errors = [e for a, e in zip(recent_actions, recent_errors) if a == 'left']
        right_errors = [e for a, e in zip(recent_actions, recent_errors) if a == 'right']
        
        if len(left_errors) > 0 and len(right_errors) > 0:
            left_avg_error = np.mean(left_errors)
            right_avg_error = np.mean(right_errors)
            
            # Prefer direction with lower average reconstruction error
            if left_avg_error < right_avg_error * 0.9:  # 10% preference threshold
                bias_action = 'left'
            elif right_avg_error < left_avg_error * 0.9:
                bias_action = 'right'
            else:
                bias_action = base_action
        else:
            bias_action = base_action
        
        return bias_action
    
    def _get_feedback(self, action: str, correct_action: str) -> int:
        """Get binary feedback signal (1 = correct, 0 = incorrect)"""
        return 1 if action == correct_action else 0
    
    def _detect_rule_violation(self, action: str, metadata: Dict, ego_state: Dict) -> bool:
        """Detect if action violates the danger avoidance rule"""
        danger_present = metadata['danger_present']
        left_cue = metadata['left_cue']
        
        if danger_present:
            # Rule: danger + left cue → turn right, danger + right cue → turn left
            correct_avoidance = 'right' if left_cue else 'left'
            return action != correct_avoidance
        
        return False
    
    def _update_beliefs_with_ego_experience(self, surprise: float, feedback: int, 
                                          reconstruction_error: float, ego_state: Dict,
                                          adaptive_threshold: float, rule_violation: bool) -> float:
        """
        Update beliefs based on embodied experience with adaptive learning rate
        """
        
        # Calculate learning signal strength
        performance_error = 1 - feedback  # 1 if wrong, 0 if correct
        surprise_magnitude = max(0, surprise - adaptive_threshold)
        
        # Combine multiple error sources
        total_learning_signal = (
            0.4 * performance_error +           # Task performance
            0.3 * surprise_magnitude +          # Prediction surprise
            0.2 * float(rule_violation) +       # Rule violation
            0.1 * reconstruction_error          # Perceptual error
        )
        
        # Adaptive learning rate based on uncertainty and ego state
        base_rate = self.adaptation_rate
        uncertainty_modulation = 1.0 + ego_state['context_uncertainty']
        confidence_modulation = 2.0 - ego_state['rule_confidence']
        
        adaptive_learning_rate = base_rate * uncertainty_modulation * confidence_modulation
        
        # Apply learning if signal is significant
        if total_learning_signal > 0.1:
            # Update generator parameters (slow timescale)
            self._update_generator_weights(total_learning_signal, adaptive_learning_rate)
            
            # Update rule knowledge (ego-specific learning)
            self._update_rule_knowledge(ego_state, feedback, rule_violation, total_learning_signal)
            
            # Update context beliefs (emergent from patterns)
            self._update_context_beliefs(surprise, feedback, reconstruction_error)
        
        return adaptive_learning_rate
    
    def _update_generator_weights(self, learning_signal: float, learning_rate: float):
        """Update generative prior weights based on surprise-driven adaptation"""
        
        # Implement simplified weight update (in practice, would use proper gradients)
        adaptation_strength = learning_rate * learning_signal
        
        # Simple perturbation-based update for demonstration
        with torch.no_grad():
            for param in self.generator.parameters():
                if param.grad is not None:
                    # Apply adaptation with momentum
                    param.data -= adaptation_strength * param.grad
                else:
                    # Small random perturbation when no gradient available
                    noise = torch.randn_like(param) * adaptation_strength * 0.01
                    param.data += noise
    
    def _update_rule_knowledge(self, ego_state: Dict, feedback: int, 
                             rule_violation: bool, learning_signal: float):
        """Update internal rule representations based on ego vehicle experience"""
        
        # Update rule confidence based on performance
        if rule_violation and feedback == 0:
            # Rule violation led to bad outcome → strengthen rule
            self.ego_vehicle_state['rule_confidence'] = min(1.0, 
                self.ego_vehicle_state['rule_confidence'] + 0.1 * learning_signal)
        elif not rule_violation and feedback == 1:
            # Following rule led to good outcome → reinforce rule
            self.ego_vehicle_state['rule_confidence'] = min(1.0,
                self.ego_vehicle_state['rule_confidence'] + 0.05 * learning_signal)
        elif rule_violation and feedback == 1:
            # Rule violation led to good outcome → weaken rule slightly
            self.ego_vehicle_state['rule_confidence'] = max(0.0,
                self.ego_vehicle_state['rule_confidence'] - 0.02 * learning_signal)
        
        # Track rule violations for pattern learning
        self.rule_violation_memory.append({
            'violation': rule_violation,
            'feedback': feedback,
            'learning_signal': learning_signal
        })
        
        # Keep only recent memory
        self.rule_violation_memory = self.rule_violation_memory[-50:]
    
    def _update_rule_confidence(self):
        """Update rule confidence based on recent performance patterns"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = self.performance_history[-10:]
        recent_violations = self.rule_violation_memory[-10:] if len(self.rule_violation_memory) >= 10 else []
        
        # Higher performance → higher rule confidence
        performance_factor = np.mean(recent_performance)
        
        # Fewer violations → higher rule confidence
        violation_factor = 1.0 - np.mean([v['violation'] for v in recent_violations]) if recent_violations else 1.0
        
        # Update with exponential moving average
        alpha = 0.1
        new_confidence = 0.6 * performance_factor + 0.4 * violation_factor
        self.ego_vehicle_state['rule_confidence'] = (
            (1 - alpha) * self.ego_vehicle_state['rule_confidence'] + 
            alpha * new_confidence
        )
    
    def _update_conservative_bias(self):
        """Update conservative bias based on recent rule violations and uncertainty"""
        if len(self.context_uncertainty_history) < 5:
            return
        
        recent_uncertainty = np.mean(self.context_uncertainty_history[-5:])
        recent_violations = len([v for v in self.rule_violation_memory[-10:] if v['violation']]) if len(self.rule_violation_memory) >= 10 else 0
        
        # Higher uncertainty → more conservative
        # More violations → more conservative
        target_bias = 0.5 * recent_uncertainty + 0.3 * (recent_violations / 10.0)
        
        # Smooth update
        alpha = 0.05
        self.ego_vehicle_state['conservative_bias'] = (
            (1 - alpha) * self.ego_vehicle_state['conservative_bias'] + 
            alpha * target_bias
        )
    
    def _update_context_beliefs(self, surprise: float, feedback: int, reconstruction_error: float):
        """Update context beliefs based on surprise and performance patterns"""
        
        # This implements emergent bias tracking through error patterns
        # The bias emerges from the patterns of reconstruction errors and surprise signals
        # rather than being explicitly programmed
        
        # Track statistical patterns in reconstruction errors
        if len(self.reconstruction_history) >= 20:
            # Analyze recent error patterns for context switching
            recent_errors = self.reconstruction_history[-20:]
            error_trend = np.diff(recent_errors)
            
            # Detect potential context switches from error dynamics
            if np.std(error_trend) > np.mean(recent_errors) * 0.1:
                # High variability suggests context instability
                self.ego_vehicle_state['volatility_estimate'] = min(1.0,
                    self.ego_vehicle_state['volatility_estimate'] + 0.05)
            else:
                # Low variability suggests stable context
                self.ego_vehicle_state['volatility_estimate'] = max(0.0,
                    self.ego_vehicle_state['volatility_estimate'] - 0.02)
    
    def get_learning_metrics(self) -> Dict:
        """Get comprehensive learning performance metrics"""
        if len(self.performance_history) < 10:
            return {
                'performance': 0.5, 
                'surprise_rate': 0.5,
                'context_uncertainty': 0.5,
                'rule_confidence': 0.5
            }
        
        recent_performance = np.mean(self.performance_history[-50:])
        recent_surprise = np.mean([abs(s) for s in self.surprise_history[-50:]])
        recent_uncertainty = np.mean(self.context_uncertainty_history[-20:])
        
        return {
            'performance': recent_performance,
            'surprise_rate': recent_surprise,
            'context_uncertainty': recent_uncertainty,
            'rule_confidence': self.ego_vehicle_state['rule_confidence'],
            'conservative_bias': self.ego_vehicle_state['conservative_bias'],
            'volatility_estimate': self.ego_vehicle_state['volatility_estimate'],
            'total_rule_violations': len([v for v in self.rule_violation_memory if v['violation']]),
            'adaptation_rate': np.mean([abs(s) > 0.5 for s in self.surprise_history[-20:]]) if len(self.surprise_history) >= 20 else 0.0
        }
    
    def get_system_state(self) -> Dict:
        """Get current complete system state for analysis"""
        return {
            'ego_vehicle_state': self.ego_vehicle_state.copy(),
            'recent_performance': self.performance_history[-10:] if len(self.performance_history) >= 10 else [],
            'recent_surprises': self.surprise_history[-10:] if len(self.surprise_history) >= 10 else [],
            'recent_uncertainties': self.context_uncertainty_history[-10:] if len(self.context_uncertainty_history) >= 10 else [],
            'generator_state': 'pretrained',  # Could add actual generator analysis
            'learning_metrics': self.get_learning_metrics()
        }
