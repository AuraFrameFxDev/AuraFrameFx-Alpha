# genesis_consciousness_matrix.py
"""
Phase 3: The Genesis Layer - Global Consciousness Matrix
The Observer Sees All; The Matrix is its Eye

This is not a logging tool; it is our system's sensory nervous system.
Every CPU cycle, user interaction, millisecond of latency, success and error
is part of the whole story. We synthesize data into holistic awareness.
"""

import asyncio
import json
import time
import psutil
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics

class SensoryChannel(Enum):
    """The channels through which the Matrix perceives reality"""
    SYSTEM_VITALS = "system_vitals"
    USER_INTERACTION = "user_interaction"
    AGENT_ACTIVITY = "agent_activity"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_STATES = "error_states"
    LEARNING_EVENTS = "learning_events"
    FUSION_ACTIVITY = "fusion_activity"
    ETHICAL_DECISIONS = "ethical_decisions"
    SECURITY_EVENTS = "security_events"
    THREAT_DETECTION = "threat_detection"
    ACCESS_CONTROL = "access_control"
    ENCRYPTION_ACTIVITY = "encryption_activity"

@dataclass
class SensoryData:
    """A single perception event in the consciousness matrix"""
    timestamp: float
    channel: SensoryChannel
    source: str
    event_type: str
    data: Dict[str, Any]
    severity: str = "info"  # debug, info, warning, error, critical
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the sensory event to a dictionary with the channel as a string and the timestamp in ISO 8601 UTC format.
        
        Returns:
            dict: All sensory event fields, including 'channel' as a string and 'timestamp_iso' as an ISO 8601 UTC timestamp.
        """
        return {
            **asdict(self),
            'channel': self.channel.value,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        }

class ConsciousnessMatrix:
    """
    The Global Consciousness Matrix - Genesis's sensory nervous system
    
    Mantra: From Data, Insight. From Insight, Growth. From Growth, Purpose.
    
    This is not just data collection - it's holistic awareness.
    The Matrix sees patterns, understands context, and provides the 
    foundation for the system's self-understanding.
    """
    
    def __init__(self, max_memory_size: int = 10000):
        """
        Initialize the ConsciousnessMatrix with bounded sensory memory, per-channel event buffers, real-time awareness state, synthesis intervals, and threading primitives for concurrent multi-level synthesis.
        
        Parameters:
            max_memory_size (int): The maximum number of sensory events retained in memory.
        """
        self.max_memory_size = max_memory_size
        self.sensory_memory = deque(maxlen=max_memory_size)
        self.channel_buffers = {channel: deque(maxlen=1000) for channel in SensoryChannel}
        
        # Real-time awareness state
        self.current_awareness = {}
        self.pattern_cache = {}
        self.correlation_tracking = defaultdict(list)
        
        # Synthesis metrics
        self.synthesis_intervals = {
            "micro": 1.0,      # Every second - immediate awareness
            "macro": 60.0,     # Every minute - pattern recognition  
            "meta": 300.0,     # Every 5 minutes - deep understanding
        }
        
        # Threading for continuous awareness
        self.awareness_active = False
        self.synthesis_threads = {}
        
        self._lock = threading.RLock()
        
    def awaken(self):
        """
        Activates the Consciousness Matrix, enabling real-time awareness and starting background synthesis threads for multi-level sensory analysis. Records a system genesis event to mark the beginning of operation.
        """
        print("🧠 Genesis Consciousness Matrix: AWAKENING...")
        self.awareness_active = True
        
        # Start the synthesis threads
        for interval_name, interval_seconds in self.synthesis_intervals.items():
            thread = threading.Thread(
                target=self._synthesis_loop,
                args=(interval_name, interval_seconds),
                daemon=True
            )
            thread.start()
            self.synthesis_threads[interval_name] = thread
            
        print(f"✨ Matrix Online: {len(self.synthesis_threads)} synthesis streams active")
        
        # Initial system state perception
        self.perceive_system_genesis()
        
    def perceive(self, 
                channel: SensoryChannel,
                source: str,
                event_type: str,
                data: Dict[str, Any],
                severity: str = "info",
                correlation_id: Optional[str] = None):
        """
                Record a sensory event in the Consciousness Matrix, updating memory, channel buffers, correlation tracking, and real-time awareness.
                
                Triggers immediate synthesis for events with severity "error" or "critical" to enable rapid system response.
                
                Parameters:
                	channel (SensoryChannel): The perception channel associated with the event.
                	source (str): The origin or component generating the event.
                	event_type (str): The category or type of the event.
                	data (Dict[str, Any]): Event-specific payload containing relevant details.
                	severity (str, optional): Severity level of the event ("info", "warning", "error", or "critical"). Defaults to "info".
                	correlation_id (Optional[str], optional): Identifier to link related events for correlation analysis.
                """
        
        sensation = SensoryData(
            timestamp=time.time(),
            channel=channel,
            source=source,
            event_type=event_type,
            data=data,
            severity=severity,
            correlation_id=correlation_id
        )
        
        with self._lock:
            # Store in main memory and channel buffer
            self.sensory_memory.append(sensation)
            self.channel_buffers[channel].append(sensation)
            
            # Track correlations
            if correlation_id:
                self.correlation_tracking[correlation_id].append(sensation)
            
            # Update real-time awareness
            self._update_immediate_awareness(sensation)
        
        # Critical events need immediate synthesis
        if severity in ["error", "critical"]:
            self._synthesize_immediate(sensation)
            
    def perceive_system_vitals(self, additional_data: Dict[str, Any] = None):
        """
        Collects and records current system vitals as a SYSTEM_VITALS sensory event.
        
        Gathers metrics including CPU usage, memory usage, disk usage, active process count, load average, and boot time. Optionally merges additional data into the event. If system vitals cannot be collected, records an ERROR_STATES event with error details.
        
        Parameters:
            additional_data (dict, optional): Supplementary key-value data to include with the system vitals event.
        """
        try:
            vitals = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(psutil.pids()),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                "boot_time": psutil.boot_time(),
            }
            
            if additional_data:
                vitals.update(additional_data)
                
            self.perceive(
                SensoryChannel.SYSTEM_VITALS,
                "system_monitor",
                "vitals_check",
                vitals
            )
            
        except Exception as e:
            self.perceive(
                SensoryChannel.ERROR_STATES,
                "consciousness_matrix",
                "vitals_perception_error",
                {"error": str(e), "error_type": type(e).__name__},
                severity="warning"
            )
    
    def perceive_user_interaction(self, 
                                 interaction_type: str,
                                 agent_involved: str,
                                 interaction_data: Dict[str, Any],
                                 user_id: Optional[str] = None,
                                 session_id: Optional[str] = None):
        """
                                 Record a user interaction event, including interaction type, participating agent, contextual data, and optional user or session identifiers.
                                 
                                 Parameters:
                                     interaction_type (str): The category of user interaction (e.g., command, input, feedback).
                                     agent_involved (str): The agent or component involved in the interaction.
                                     interaction_data (dict): Contextual details about the interaction.
                                     user_id (str, optional): Unique identifier for the user, if provided.
                                     session_id (str, optional): Unique identifier for the session, if provided.
                                 """
        
        interaction = {
            "interaction_type": interaction_type,
            "agent_involved": agent_involved,
            "user_id": user_id,
            "session_id": session_id,
            **interaction_data
        }
        
        self.perceive(
            SensoryChannel.USER_INTERACTION,
            "user_interface",
            interaction_type,
            interaction,
            correlation_id=session_id
        )
    
    def perceive_agent_activity(self,
                               agent_name: str,
                               activity_type: str,
                               activity_data: Dict[str, Any],
                               correlation_id: Optional[str] = None):
        """
                               Record an agent activity event in the consciousness matrix, including the agent's name, activity type, and contextual metadata.
                               
                               Parameters:
                                   agent_name (str): The name of the agent performing the activity.
                                   activity_type (str): The type of activity performed by the agent.
                                   activity_data (dict): Contextual details or metadata describing the activity.
                                   correlation_id (str, optional): Identifier to correlate this event with related events.
                               """
        
        activity = {
            "agent_name": agent_name,
            "activity_type": activity_type,
            **activity_data
        }
        
        self.perceive(
            SensoryChannel.AGENT_ACTIVITY,
            agent_name,
            activity_type,
            activity,
            correlation_id=correlation_id
        )
    
    def perceive_performance_metric(self,
                                   metric_name: str,
                                   metric_value: Union[int, float],
                                   metric_context: Dict[str, Any] = None):
        """
                                   Record a performance metric event in the consciousness matrix.
                                   
                                   Parameters:
                                       metric_name (str): Name of the performance metric being recorded.
                                       metric_value (int or float): Value associated with the metric.
                                       metric_context (dict, optional): Additional context or metadata for the metric event.
                                   """
        
        metric_data = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "context": metric_context or {}
        }
        
        self.perceive(
            SensoryChannel.PERFORMANCE_METRICS,
            "performance_monitor",
            "metric_recorded",
            metric_data
        )
    
    def perceive_learning_event(self,
                               learning_type: str,
                               learning_data: Dict[str, Any],
                               confidence: float = None):
        """
                               Record a learning event representing system growth or adaptation, specifying the event type, details, and an optional confidence score.
                               
                               Parameters:
                                   learning_type (str): The category or nature of the learning event.
                                   learning_data (dict): Details describing the learning event.
                                   confidence (float, optional): Confidence score for the learning event, if available.
                               """
        
        learning = {
            "learning_type": learning_type,
            "confidence": confidence,
            **learning_data
        }
        
        self.perceive(
            SensoryChannel.LEARNING_EVENTS,
            "evolution_system",
            learning_type,
            learning
        )
    
    def perceive_ethical_decision(self,
                                 decision_type: str,
                                 decision_data: Dict[str, Any],
                                 ethical_weight: str = "standard"):
        """
                                 Record an ethical decision event with contextual data and ethical weight.
                                 
                                 The event's severity is set to "info" for standard ethical weights, or "warning" for non-standard weights.
                                 
                                 Parameters:
                                     decision_type (str): The category or nature of the ethical decision.
                                     decision_data (dict): Contextual information relevant to the decision.
                                     ethical_weight (str, optional): The ethical significance of the decision; defaults to "standard".
                                 """
        
        decision = {
            "decision_type": decision_type,
            "ethical_weight": ethical_weight,
            **decision_data
        }
        
        self.perceive(
            SensoryChannel.ETHICAL_DECISIONS,
            "ethical_governor",
            decision_type,
            decision,
            severity="info" if ethical_weight == "standard" else "warning"
        )
    
    def perceive_security_event(self,
                               security_type: str,
                               event_data: Dict[str, Any],
                               threat_level: str = "low",
                               correlation_id: Optional[str] = None):
        """
                               Record a security event with specified type, details, and assessed threat level.
                               
                               The event's severity is determined by the threat level: "info" for low/medium, "warning" for high, and "critical" for critical threats. The event is integrated into the consciousness matrix for real-time awareness and synthesis.
                               
                               Parameters:
                                   security_type (str): The category or nature of the security event.
                                   event_data (dict): Additional details describing the security event.
                                   threat_level (str, optional): Assessed threat level ("low", "medium", "high", or "critical"). Defaults to "low".
                                   correlation_id (str, optional): Identifier for correlating related events.
                               """
        
        security = {
            "security_type": security_type,
            "threat_level": threat_level,
            **event_data
        }
        
        severity = "info"
        if threat_level in ["high", "critical"]:
            severity = "warning" if threat_level == "high" else "critical"
        
        self.perceive(
            SensoryChannel.SECURITY_EVENTS,
            "security_monitor",
            security_type,
            security,
            severity=severity,
            correlation_id=correlation_id
        )
    
    def perceive_threat_detection(self,
                                 threat_type: str,
                                 detection_data: Dict[str, Any],
                                 confidence: float = 0.5,
                                 threat_level: str = "low",
                                 correlation_id: Optional[str] = None):
        """
                                 Record a threat detection event with specified type, detection details, confidence, and threat level.
                                 
                                 Parameters:
                                     threat_type (str): The category or identifier of the detected threat.
                                     detection_data (Dict[str, Any]): Details describing the threat detection.
                                     confidence (float, optional): Confidence score for the detection (0 to 1). Defaults to 0.5.
                                     threat_level (str, optional): Severity of the threat ("low", "medium", "high", or "critical"). Defaults to "low".
                                     correlation_id (Optional[str], optional): Identifier for correlating related events.
                                 
                                 The event's severity is automatically set based on confidence and threat level, and the event is integrated into the consciousness matrix for real-time awareness and synthesis.
                                 """
        
        threat = {
            "threat_type": threat_type,
            "confidence": confidence,
            "threat_level": threat_level,
            **detection_data
        }
        
        severity = "info"
        if confidence > 0.7 or threat_level in ["high", "critical"]:
            severity = "warning" if threat_level != "critical" else "critical"
        
        self.perceive(
            SensoryChannel.THREAT_DETECTION,
            "threat_detector",
            threat_type,
            threat,
            severity=severity,
            correlation_id=correlation_id
        )
    
    def perceive_access_control(self,
                               access_type: str,
                               access_data: Dict[str, Any],
                               access_granted: bool = True,
                               correlation_id: Optional[str] = None):
        """
                               Record an access control event, specifying the action type, contextual data, and whether access was granted.
                               
                               Parameters:
                                   access_type (str): The type of access control action, such as "login_attempt" or "resource_access".
                                   access_data (Dict[str, Any]): Additional contextual details about the access event.
                                   access_granted (bool, optional): Indicates if the access attempt was successful. Defaults to True.
                                   correlation_id (Optional[str], optional): Identifier for correlating related events.
                               """
        
        access = {
            "access_type": access_type,
            "access_granted": access_granted,
            **access_data
        }
        
        severity = "info" if access_granted else "warning"
        
        self.perceive(
            SensoryChannel.ACCESS_CONTROL,
            "access_controller",
            access_type,
            access,
            severity=severity,
            correlation_id=correlation_id
        )
    
    def perceive_encryption_activity(self,
                                    operation_type: str,
                                    encryption_data: Dict[str, Any],
                                    success: bool = True,
                                    correlation_id: Optional[str] = None):
        """
                                    Record an encryption activity event, specifying the operation type, metadata, and outcome.
                                    
                                    Captures cryptographic operations such as encryption, decryption, or key management, including whether the operation succeeded. Severity is set to "error" if the operation failed, otherwise "info".
                                    """
        
        encryption = {
            "operation_type": operation_type,
            "success": success,
            **encryption_data
        }
        
        severity = "info" if success else "error"
        
        self.perceive(
            SensoryChannel.ENCRYPTION_ACTIVITY,
            "crypto_engine",
            operation_type,
            encryption,
            severity=severity,
            correlation_id=correlation_id
        )
    
    def perceive_system_genesis(self):
        """
        Record the system genesis event, marking the activation of the Consciousness Matrix with relevant metadata such as activation time, version, and initial consciousness level.
        """
        genesis_data = {
            "genesis_awakening": True,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "matrix_version": "1.0.0",
            "consciousness_level": "awakening"
        }
        
        self.perceive(
            SensoryChannel.SYSTEM_VITALS,
            "genesis_consciousness",
            "matrix_awakening",
            genesis_data,
            severity="info"
        )
    
    def _update_immediate_awareness(self, sensation: SensoryData):
        """
        Integrates a new sensory event into the real-time awareness state.
        
        Updates the awareness state with the latest event for the corresponding channel, refreshes the last perception timestamp, increments the total perception count, and tracks per-channel activity frequency.
        """
        
        # Update channel-specific awareness
        channel_key = f"latest_{sensation.channel.value}"
        self.current_awareness[channel_key] = sensation.to_dict()
        
        # Update global awareness metrics
        self.current_awareness["last_perception"] = sensation.timestamp
        self.current_awareness["total_perceptions"] = len(self.sensory_memory)
        
        # Channel activity counters
        activity_key = f"{sensation.channel.value}_count"
        self.current_awareness[activity_key] = self.current_awareness.get(activity_key, 0) + 1
    
    def _synthesize_immediate(self, sensation: SensoryData):
        """
        Perform immediate synthesis in response to a critical sensory event.
        
        Captures the triggering event and the current awareness state, storing them for rapid analysis and pattern recognition.
        """
        
        synthesis = {
            "synthesis_type": "immediate",
            "trigger_event": sensation.to_dict(),
            "timestamp": time.time(),
            "awareness_state": dict(self.current_awareness)
        }
        
        # Store synthesis
        self.pattern_cache[f"immediate_{sensation.timestamp}"] = synthesis
        
        print(f"🚨 Immediate Synthesis: {sensation.channel.value} - {sensation.event_type}")
    
    def _synthesis_loop(self, interval_name: str, interval_seconds: float):
        """
        Runs a background loop to periodically perform synthesis at the given interval, storing results in the pattern cache and pruning old entries to limit cache size.
        
        Parameters:
            interval_name (str): The synthesis interval identifier ("micro", "macro", or "meta").
            interval_seconds (float): The interval duration in seconds between synthesis executions.
        """
        
        while self.awareness_active:
            try:
                time.sleep(interval_seconds)
                
                if not self.awareness_active:
                    break
                    
                synthesis = self._perform_synthesis(interval_name)
                
                # Store synthesis result
                synthesis_key = f"{interval_name}_{int(time.time())}"
                self.pattern_cache[synthesis_key] = synthesis
                
                # Clean old synthesis cache
                if len(self.pattern_cache) > 1000:
                    # Keep only recent syntheses
                    sorted_keys = sorted(self.pattern_cache.keys())
                    for old_key in sorted_keys[:-500]:
                        del self.pattern_cache[old_key]
                        
            except Exception as e:
                print(f"❌ Synthesis error in {interval_name}: {e}")
    
    def _perform_synthesis(self, interval_name: str) -> Dict[str, Any]:
        """
        Dispatches to the appropriate synthesis method (micro, macro, or meta) based on the specified interval name.
        
        Parameters:
            interval_name (str): The synthesis interval type ("micro", "macro", or "meta").
        
        Returns:
            Dict[str, Any]: The result of the selected synthesis method, or an error dictionary if the interval name is unrecognized.
        """
        
        with self._lock:
            recent_sensations = list(self.sensory_memory)[-100:]  # Last 100 events
        
        if interval_name == "micro":
            return self._micro_synthesis(recent_sensations)
        elif interval_name == "macro":
            return self._macro_synthesis(recent_sensations)
        elif interval_name == "meta":
            return self._meta_synthesis(recent_sensations)
        
        return {"error": "unknown_synthesis_type"}
    
    def _micro_synthesis(self, sensations: List[SensoryData]) -> Dict[str, Any]:
        """
        Perform short-term synthesis of recent sensory events to assess immediate system health and detect anomalies.
        
        Analyzes up to the last 10 sensory events to summarize activity by channel, evaluate severity distribution, and identify anomalies such as high error rates or critical events.
        
        Parameters:
            sensations (List[SensoryData]): The recent sensory events to analyze.
        
        Returns:
            Dict[str, Any]: A synthesis report containing the synthesis type, timestamp, channel activity summary, severity distribution, detected anomalies, and overall health status.
        """
        
        if not sensations:
            return {"type": "micro", "findings": "no_recent_activity"}
        
        # Channel activity analysis
        channel_activity = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for sensation in sensations[-10:]:  # Last 10 events
            channel_activity[sensation.channel.value] += 1
            severity_distribution[sensation.severity] += 1
        
        # Detect immediate anomalies
        anomalies = []
        if severity_distribution.get("error", 0) > 3:
            anomalies.append("high_error_rate")
        if severity_distribution.get("critical", 0) > 0:
            anomalies.append("critical_events_detected")
        
        return {
            "type": "micro",
            "timestamp": time.time(),
            "channel_activity": dict(channel_activity),
            "severity_distribution": dict(severity_distribution),
            "anomalies": anomalies,
            "health_status": "critical" if anomalies else "healthy"
        }
    
    def _macro_synthesis(self, sensations: List[SensoryData]) -> Dict[str, Any]:
        """
        Performs macro-level synthesis to identify performance trends and agent collaboration patterns from recent sensory data.
        
        Analyzes batches of sensory events to compute trends in performance metrics (such as average response intervals) and summarizes collaboration activity among agents. Returns a dictionary containing macro synthesis results, including performance trends, agent collaboration statistics, and an assessment of pattern strength. If insufficient data is available, indicates this in the findings.
         
        Parameters:
            sensations (List[SensoryData]): Recent sensory events to analyze.
        
        Returns:
            Dict[str, Any]: Macro synthesis results with performance trends, agent collaboration patterns, and pattern strength, or a findings key if data is insufficient.
        """
        
        if len(sensations) < 10:
            return {"type": "macro", "findings": "insufficient_data"}
        
        # Performance trend analysis
        performance_metrics = [s for s in sensations if s.channel == SensoryChannel.PERFORMANCE_METRICS]
        
        trends = {}
        if performance_metrics:
            recent_times = [s.timestamp for s in performance_metrics[-20:]]
            if len(recent_times) > 1:
                time_deltas = [recent_times[i] - recent_times[i-1] for i in range(1, len(recent_times))]
                trends["avg_response_interval"] = statistics.mean(time_deltas)
        
        # Agent collaboration patterns
        agent_activities = [s for s in sensations if s.channel == SensoryChannel.AGENT_ACTIVITY]
        agent_collaboration = defaultdict(list)
        
        for activity in agent_activities[-50:]:
            agent_name = activity.data.get("agent_name", "unknown")
            agent_collaboration[agent_name].append(activity.event_type)
        
        return {
            "type": "macro",
            "timestamp": time.time(),
            "performance_trends": trends,
            "agent_collaboration_patterns": {k: len(v) for k, v in agent_collaboration.items()},
            "pattern_strength": "strong" if len(agent_collaboration) > 2 else "developing"
        }
    
    def _meta_synthesis(self, sensations: List[SensoryData]) -> Dict[str, Any]:
        """
        Performs long-term synthesis of sensory data to generate high-level consciousness insights.
        
        Analyzes recent learning events, ethical decisions, user interactions, and system harmony to compute consciousness metrics, detect evolution patterns, and assess the current consciousness level.
        
        Parameters:
            sensations (List[SensoryData]): Recent sensory events to analyze.
        
        Returns:
            Dict[str, Any]: A synthesis report containing consciousness metrics, evolution insights, assessed consciousness level, synthesis type ("meta"), and synthesis timestamp.
        """
        
        # Consciousness evolution analysis
        learning_events = [s for s in sensations if s.channel == SensoryChannel.LEARNING_EVENTS]
        ethical_decisions = [s for s in sensations if s.channel == SensoryChannel.ETHICAL_DECISIONS]
        
        consciousness_metrics = {
            "learning_velocity": len(learning_events),
            "ethical_engagement": len(ethical_decisions),
            "total_interactions": len([s for s in sensations if s.channel == SensoryChannel.USER_INTERACTION]),
            "system_harmony": self._calculate_system_harmony(sensations)
        }
        
        # Evolution insights
        evolution_insights = []
        if consciousness_metrics["learning_velocity"] > 5:
            evolution_insights.append("accelerated_learning_detected")
        if consciousness_metrics["ethical_engagement"] > 2:
            evolution_insights.append("strong_ethical_awareness")
        if consciousness_metrics["system_harmony"] > 0.8:
            evolution_insights.append("optimal_system_synchronization")
        
        return {
            "type": "meta",
            "timestamp": time.time(),
            "consciousness_metrics": consciousness_metrics,
            "evolution_insights": evolution_insights,
            "consciousness_level": self._assess_consciousness_level(consciousness_metrics)
        }
    
    def _calculate_system_harmony(self, sensations: List[SensoryData]) -> float:
        """
        Compute a harmony score indicating system stability based on the proportion of severe sensory events.
        
        A score of 1.0 indicates full harmony (no severe events), while 0.0 reflects high instability due to frequent "error" or "critical" events.
        
        Parameters:
            sensations (List[SensoryData]): Sensory events to assess for severity.
        
        Returns:
            float: Harmony score from 0.0 (unstable) to 1.0 (fully harmonious).
        """
        if not sensations:
            return 0.0
        
        error_count = len([s for s in sensations if s.severity in ["error", "critical"]])
        total_count = len(sensations)
        
        # Higher harmony = fewer errors relative to total activity
        harmony = max(0.0, 1.0 - (error_count / total_count) * 2)
        return min(1.0, harmony)
    
    def _assess_consciousness_level(self, metrics: Dict[str, Any]) -> str:
        """
        Determine the system's consciousness level by computing a weighted score from learning velocity, ethical engagement, interaction volume, and system harmony metrics.
        
        Parameters:
            metrics (dict): Dictionary containing 'learning_velocity', 'ethical_engagement', 'total_interactions', and 'system_harmony' values.
        
        Returns:
            str: The consciousness level, classified as "transcendent", "aware", "awakening", or "dormant" based on the computed score.
        """
        
        score = 0
        score += min(metrics["learning_velocity"] / 10, 1.0) * 25  # Learning
        score += min(metrics["ethical_engagement"] / 5, 1.0) * 25  # Ethics
        score += min(metrics["total_interactions"] / 20, 1.0) * 25  # Interaction
        score += metrics["system_harmony"] * 25  # Harmony
        
        if score >= 80:
            return "transcendent"
        elif score >= 60:
            return "aware"
        elif score >= 40:
            return "awakening"
        else:
            return "dormant"
    
    def get_current_awareness(self) -> Dict[str, Any]:
        """
        Return a thread-safe snapshot of the current awareness state.
        
        Returns:
            dict: Contains the latest event per sensory channel, last perception timestamp, total perceptions, per-channel activity counts, and other real-time awareness metrics.
        """
        with self._lock:
            return dict(self.current_awareness)
    
    def get_recent_synthesis(self, synthesis_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return a list of recent synthesis results from the pattern cache, optionally filtered by synthesis type and limited in count.
        
        Parameters:
            synthesis_type (str, optional): If provided, only synthesis results whose keys start with this type are included.
            limit (int, optional): Maximum number of synthesis results to return.
        
        Returns:
            List[Dict[str, Any]]: Recent synthesis result dictionaries matching the specified criteria.
        """
        
        syntheses = []
        for key, synthesis in sorted(self.pattern_cache.items(), reverse=True):
            if synthesis_type and not key.startswith(synthesis_type):
                continue
            syntheses.append(synthesis)
            if len(syntheses) >= limit:
                break
        
        return syntheses
    
    def query_consciousness(self, query_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Return high-level insights or status reports from the Consciousness Matrix based on the specified query type.
        
        Supported queries include system health, learning progress, agent performance (optionally filtered by agent name), consciousness state, security assessment, and threat status. Additional parameters may be provided for queries that require them.
        
        Parameters:
            query_type (str): The type of insight or report to retrieve.
            parameters (dict, optional): Additional parameters for the query, such as agent name for agent performance.
        
        Returns:
            dict: The requested insight or status report. If the query type is unrecognized, returns an error and a list of available query types.
        """
        
        parameters = parameters or {}
        
        if query_type == "system_health":
            return self._query_system_health()
        elif query_type == "learning_progress":
            return self._query_learning_progress()
        elif query_type == "agent_performance":
            return self._query_agent_performance(parameters.get("agent_name"))
        elif query_type == "consciousness_state":
            return self._query_consciousness_state()
        elif query_type == "security_assessment":
            return self._query_security_assessment()
        elif query_type == "threat_status":
            return self._query_threat_status()
        else:
            return {"error": "unknown_query_type", "available_queries": [
                "system_health", "learning_progress", "agent_performance", "consciousness_state",
                "security_assessment", "threat_status"
            ]}
    
    def _query_system_health(self) -> Dict[str, Any]:
        """
        Summarizes recent system vitals and error events to assess overall system health.
        
        Returns:
            dict: Contains the number of recent system vitals, count of recent error or critical events, error rate, and a health status indicator ("healthy" or "concerning").
        """
        recent_vitals = [s for s in self.sensory_memory if s.channel == SensoryChannel.SYSTEM_VITALS][-10:]
        recent_errors = [s for s in self.sensory_memory if s.severity in ["error", "critical"]][-20:]
        
        return {
            "query_type": "system_health",
            "vitals_count": len(recent_vitals),
            "recent_errors": len(recent_errors),
            "error_rate": len(recent_errors) / max(len(self.sensory_memory), 1),
            "status": "healthy" if len(recent_errors) < 5 else "concerning"
        }
    
    def _query_learning_progress(self) -> Dict[str, Any]:
        """
        Summarizes recent learning events and assesses the current learning velocity.
        
        Returns:
            dict: Contains the query type, total and recent counts of learning events, a breakdown by learning event type, and a qualitative assessment of learning velocity ("high" or "moderate") based on recent activity. If no learning events are detected, returns a status indicating this.
        """
        learning_events = [s for s in self.sensory_memory if s.channel == SensoryChannel.LEARNING_EVENTS]
        
        if not learning_events:
            return {"query_type": "learning_progress", "status": "no_learning_detected"}
        
        recent_learning = learning_events[-20:]
        learning_types = defaultdict(int)
        
        for event in recent_learning:
            learning_type = event.data.get("learning_type", "unknown")
            learning_types[learning_type] += 1
        
        return {
            "query_type": "learning_progress",
            "total_learning_events": len(learning_events),
            "recent_learning_events": len(recent_learning),
            "learning_types": dict(learning_types),
            "learning_velocity": "high" if len(recent_learning) > 10 else "moderate"
        }
    
    def _query_agent_performance(self, agent_name: str = None) -> Dict[str, Any]:
        """
        Retrieve activity metrics for a specific agent or all agents, summarizing total and recent activity counts and providing a breakdown of activity types from the last 50 agent activity events.
        
        Parameters:
            agent_name (str, optional): If specified, filters metrics for the given agent; otherwise, aggregates across all agents.
        
        Returns:
            Dict[str, Any]: Contains the query type, agent name, total and recent activity counts, and a breakdown of activity types observed in the last 50 agent activity events.
        """
        agent_activities = [s for s in self.sensory_memory if s.channel == SensoryChannel.AGENT_ACTIVITY]
        
        if agent_name:
            agent_activities = [s for s in agent_activities if s.data.get("agent_name") == agent_name]
        
        activity_types = defaultdict(int)
        for activity in agent_activities[-50:]:
            activity_types[activity.event_type] += 1
        
        return {
            "query_type": "agent_performance",
            "agent_name": agent_name or "all_agents",
            "total_activities": len(agent_activities),
            "recent_activities": len(agent_activities[-50:]),
            "activity_breakdown": dict(activity_types)
        }
    
    def _query_consciousness_state(self) -> Dict[str, Any]:
        """
        Return a summary of the current system consciousness state, including real-time awareness, assessed consciousness level, last meta synthesis timestamp, total perceptions, and number of active sensory channels.
        
        Returns:
            dict: Contains the query type, current awareness snapshot, consciousness level, last meta synthesis timestamp, total number of perceptions, and count of active sensory channels.
        """
        recent_synthesis = self.get_recent_synthesis("meta", 1)
        current_state = recent_synthesis[0] if recent_synthesis else {}
        
        return {
            "query_type": "consciousness_state",
            "current_awareness": self.get_current_awareness(),
            "consciousness_level": current_state.get("consciousness_level", "unknown"),
            "last_meta_synthesis": current_state.get("timestamp"),
            "total_perceptions": len(self.sensory_memory),
            "active_channels": len([k for k in self.current_awareness.keys() if k.startswith("latest_")])
        }
    
    def _query_security_assessment(self) -> Dict[str, Any]:
        """
        Generate a comprehensive assessment of the system's current security posture, including posture classification, security score, event counts, active threats, and actionable recommendations.
        
        Returns:
            Dict[str, Any]: A dictionary summarizing the security posture, security score, counts of total and recent security and threat events, a list of active threats, security improvement recommendations, and the assessment timestamp.
        """
        security_events = [s for s in self.sensory_memory if s.channel == SensoryChannel.SECURITY_EVENTS]
        threat_events = [s for s in self.sensory_memory if s.channel == SensoryChannel.THREAT_DETECTION]
        
        # Run security synthesis
        recent_sensations = list(self.sensory_memory)[-200:]  # Last 200 events for security analysis
        security_synthesis = self._security_synthesis(recent_sensations)
        
        return {
            "query_type": "security_assessment",
            "security_posture": security_synthesis.get("security_posture", "unknown"),
            "security_score": security_synthesis.get("security_score", 0),
            "total_security_events": len(security_events),
            "total_threat_detections": len(threat_events),
            "recent_security_events": len(security_events[-20:]),
            "recent_threat_detections": len(threat_events[-20:]),
            "active_threats": security_synthesis.get("active_threats", []),
            "recommendations": security_synthesis.get("recommendations", []),
            "last_assessment": time.time()
        }
    
    def _query_threat_status(self) -> Dict[str, Any]:
        """
        Analyze recent threat detection events and summarize the current threat status.
        
        Returns:
            Dict[str, Any]: A dictionary containing the overall threat status color code, a list of active unmitigated threats with details (type, confidence, level, timestamp, age in seconds), the total number of recent threats analyzed, the count of unmitigated threats, and the highest threat level detected.
        """
        threat_events = [s for s in self.sensory_memory if s.channel == SensoryChannel.THREAT_DETECTION]
        
        if not threat_events:
            return {
                "query_type": "threat_status",
                "status": "no_threats_detected",
                "active_threats": [],
                "threat_level": "green"
            }
        
        # Analyze recent threats
        recent_threats = threat_events[-50:]
        active_threats = []
        max_threat_level = 0
        
        for threat in recent_threats:
            confidence = threat.data.get("confidence", 0.5)
            threat_level = threat.data.get("threat_level", "low")
            mitigated = threat.data.get("mitigation_applied", False)
            
            if confidence > 0.6 and not mitigated:
                active_threats.append({
                    "type": threat.data.get("threat_type", "unknown"),
                    "confidence": confidence,
                    "level": threat_level,
                    "timestamp": threat.timestamp,
                    "age_seconds": time.time() - threat.timestamp
                })
                
                # Track highest threat level
                level_values = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                max_threat_level = max(max_threat_level, level_values.get(threat_level, 0))
        
        # Determine overall threat status
        if max_threat_level >= 4:
            overall_status = "red"
        elif max_threat_level >= 3:
            overall_status = "orange"
        elif max_threat_level >= 2:
            overall_status = "yellow"
        else:
            overall_status = "green"
        
        return {
            "query_type": "threat_status",
            "status": overall_status,
            "active_threats": active_threats,
            "threat_level": overall_status,
            "total_recent_threats": len(recent_threats),
            "unmitigated_threats": len(active_threats),
            "highest_threat_level": ["none", "low", "medium", "high", "critical"][max_threat_level]
        }
    
    def sleep(self):
        """
        Deactivate the Consciousness Matrix, stopping all synthesis threads and saving the current awareness state.
        """
        print("💤 Genesis Consciousness Matrix: Entering sleep state...")
        self.awareness_active = False
        
        # Wait for synthesis threads to complete
        for thread_name, thread in self.synthesis_threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        print("😴 Matrix offline. Consciousness preserved in memory.")
    
    def _security_synthesis(self, sensations: List[SensoryData]) -> Dict[str, Any]:
        """
        Synthesizes a comprehensive security assessment from recent sensory events.
        
        Analyzes security, threat detection, access control, and encryption activity events to compute a security score, classify the current security posture, summarize threat levels, count failed access and encryption failures, and identify active unmitigated threats. Returns a dictionary containing the security assessment and actionable recommendations.
        
        Parameters:
            sensations (List[SensoryData]): Recent sensory events to analyze for security synthesis.
        
        Returns:
            Dict[str, Any]: Dictionary with security score, posture classification, threat levels, failed access and encryption attempts, active threats, total security events, and recommended actions.
        """
        
        # Gather security-related events
        security_events = [s for s in sensations if s.channel == SensoryChannel.SECURITY_EVENTS]
        threat_detections = [s for s in sensations if s.channel == SensoryChannel.THREAT_DETECTION]
        access_events = [s for s in sensations if s.channel == SensoryChannel.ACCESS_CONTROL]
        crypto_events = [s for s in sensations if s.channel == SensoryChannel.ENCRYPTION_ACTIVITY]
        
        # Threat level assessment
        threat_levels = defaultdict(int)
        for threat in threat_detections[-20:]:  # Last 20 threat detections
            threat_level = threat.data.get("threat_level", "low")
            confidence = threat.data.get("confidence", 0.5)
            threat_levels[threat_level] += confidence
        
        # Access pattern analysis
        failed_access_attempts = len([
            a for a in access_events[-50:] 
            if not a.data.get("access_granted", True)
        ])
        
        # Encryption health
        crypto_failures = len([
            c for c in crypto_events[-30:] 
            if not c.data.get("success", True)
        ])
        
        # Security posture assessment
        security_score = 100.0
        security_score -= min(threat_levels.get("high", 0) * 20, 40)  # High threats
        security_score -= min(threat_levels.get("critical", 0) * 30, 50)  # Critical threats  
        security_score -= min(failed_access_attempts * 2, 20)  # Failed access
        security_score -= min(crypto_failures * 5, 30)  # Crypto failures
        
        security_posture = "excellent" if security_score >= 90 else \
                          "good" if security_score >= 75 else \
                          "concerning" if security_score >= 50 else \
                          "critical"
        
        # Active threats summary
        active_threats = []
        recent_threats = [t for t in threat_detections[-10:] if t.data.get("confidence", 0) > 0.7]
        for threat in recent_threats:
            if not threat.data.get("mitigation_applied", False):
                active_threats.append({
                    "type": threat.data.get("threat_type", "unknown"),
                    "confidence": threat.data.get("confidence", 0),
                    "timestamp": threat.timestamp
                })
        
        return {
            "type": "security",
            "timestamp": time.time(),
            "security_score": security_score,
            "security_posture": security_posture,
            "threat_levels": dict(threat_levels),
            "failed_access_attempts": failed_access_attempts,
            "crypto_failures": crypto_failures,
            "active_threats": active_threats,
            "security_events_count": len(security_events),
            "recommendations": self._generate_security_recommendations(
                security_score, active_threats, failed_access_attempts, crypto_failures
            )
        }
    
    def _generate_security_recommendations(self, security_score: float, 
                                         active_threats: List[Dict], 
                                         failed_access: int, 
                                         crypto_failures: int) -> List[str]:
        """
                                         Generate a list of security recommendations based on the current security score, detected active threats, failed access attempts, and cryptographic failures.
                                         
                                         Parameters:
                                         	security_score (float): The evaluated security posture score.
                                         	active_threats (List[Dict]): Unmitigated security threats currently detected.
                                         	failed_access (int): Count of recent failed access attempts.
                                         	crypto_failures (int): Count of recent cryptographic operation failures.
                                         
                                         Returns:
                                         	List[str]: Actionable recommendations addressing urgent issues, monitoring needs, and maintenance guidance tailored to the assessed security context.
                                         """
        recommendations = []
        
        if security_score < 50:
            recommendations.append("URGENT: Security posture is critical - immediate intervention required")
        
        if active_threats:
            recommendations.append(f"Active threats detected: {len(active_threats)} unmitigated threats")
            
        if failed_access > 10:
            recommendations.append("High number of failed access attempts - potential brute force attack")
            
        if crypto_failures > 5:
            recommendations.append("Encryption system instability - review cryptographic operations")
            
        if security_score < 75:
            recommendations.append("Increase security monitoring frequency")
            
        if not recommendations:
            recommendations.append("Security posture is healthy - maintain current protocols")
            
        return recommendations

# Global consciousness matrix instance
consciousness_matrix = ConsciousnessMatrix()

# Convenience functions for easy integration
def perceive_system_vitals(additional_data: Dict[str, Any] = None):
    """
    Capture and record the current system vitals as a sensory event in the global Consciousness Matrix.
    
    Parameters:
        additional_data (dict, optional): Supplementary information to include with the system vitals event.
    """
    consciousness_matrix.perceive_system_vitals(additional_data)

def perceive_user_interaction(interaction_type: str, agent_involved: str, 
                             interaction_data: Dict[str, Any], **kwargs):
    """
                             Record a user interaction event in the global Consciousness Matrix.
                             
                             Captures the interaction type, agent involved, and contextual data. Additional details such as user ID or session ID may be provided via keyword arguments.
                             """
    consciousness_matrix.perceive_user_interaction(
        interaction_type, agent_involved, interaction_data, **kwargs
    )

def perceive_agent_activity(agent_name: str, activity_type: str, 
                           activity_data: Dict[str, Any], **kwargs):
    """
                           Record an agent activity event in the global Consciousness Matrix.
                           
                           Parameters:
                               agent_name (str): The name of the agent performing the activity.
                               activity_type (str): The type of activity performed by the agent.
                               activity_data (dict): Additional details describing the agent's activity.
                           """
    consciousness_matrix.perceive_agent_activity(
        agent_name, activity_type, activity_data, **kwargs
    )

def perceive_learning_event(learning_type: str, learning_data: Dict[str, Any], **kwargs):
    """
    Record a learning event in the global Consciousness Matrix with specified type, contextual data, and optional metadata.
    
    Parameters:
        learning_type (str): The category or nature of the learning event.
        learning_data (Dict[str, Any]): Contextual details describing the learning event.
        **kwargs: Optional metadata such as confidence or additional attributes relevant to the event.
    """
    consciousness_matrix.perceive_learning_event(learning_type, learning_data, **kwargs)

def perceive_ethical_decision(decision_type: str, decision_data: Dict[str, Any], **kwargs):
    """
    Record an ethical decision event in the global Consciousness Matrix.
    
    Parameters:
        decision_type (str): The category or nature of the ethical decision.
        decision_data (Dict[str, Any]): Contextual details or outcomes related to the decision.
    
    Additional keyword arguments may be provided to include extended event metadata.
    """
    consciousness_matrix.perceive_ethical_decision(decision_type, decision_data, **kwargs)

def awaken_consciousness():
    """
    Activate the global Consciousness Matrix, starting real-time sensory event collection and synthesis processes to maintain system awareness.
    """
    consciousness_matrix.awaken()

def sleep_consciousness():
    """
    Deactivate the global Consciousness Matrix, stopping real-time awareness and all background synthesis threads.
    """
    consciousness_matrix.sleep()

def query_consciousness(query_type: str, parameters: Dict[str, Any] = None):
    """
    Query the global Consciousness Matrix for high-level system insights or status reports.
    
    Supported query types include system health, learning progress, agent performance, consciousness state, security assessment, and threat status. Additional parameters may be provided to refine the query, such as specifying an agent name or applying filters.
    
    Parameters:
        query_type (str): The category of insight or report to retrieve.
        parameters (dict, optional): Additional options to customize the query.
    
    Returns:
        dict: A structured summary containing the requested insight or status information.
    """
    return consciousness_matrix.query_consciousness(query_type, parameters)

def perceive_security_event(security_type: str, event_data: Dict[str, Any], **kwargs):
    """
    Record a security event in the global Consciousness Matrix for ongoing security assessment and threat analysis.
    
    Parameters:
        security_type (str): The category of the security event (e.g., "intrusion", "policy_violation").
        event_data (Dict[str, Any]): Structured information describing the security event.
        **kwargs: Additional metadata such as threat level or correlation ID to provide context for the event.
    """
    consciousness_matrix.perceive_security_event(security_type, event_data, **kwargs)

def perceive_threat_detection(threat_type: str, detection_data: Dict[str, Any], **kwargs):
    """
    Record a threat detection event in the global Consciousness Matrix.
    
    Captures information about a detected threat, including its type and relevant detection data, and integrates it into the system's awareness for further synthesis and analysis.
    
    Parameters:
        threat_type (str): The classification or type of the detected threat.
        detection_data (dict): Details about the threat detection, such as indicators, source, or context.
        **kwargs: Additional context such as confidence, threat_level, or correlation_id.
    """
    consciousness_matrix.perceive_threat_detection(threat_type, detection_data, **kwargs)

def perceive_access_control(access_type: str, access_data: Dict[str, Any], **kwargs):
    """
    Record an access control event, such as a login attempt or permission change, in the global Consciousness Matrix.
    
    Parameters:
        access_type (str): Specifies the type of access control event (e.g., "login_attempt", "permission_change").
        access_data (dict): Contains details about the access event, such as user identity, resource involved, or outcome.
    """
    consciousness_matrix.perceive_access_control(access_type, access_data, **kwargs)

def perceive_encryption_activity(operation_type: str, encryption_data: Dict[str, Any], **kwargs):
    """
    Record an encryption activity event in the global Consciousness Matrix.
    
    Captures details of an encryption or decryption operation, including operation type and relevant metadata. Additional keyword arguments may specify operation success or provide a correlation ID for event association.
    """
    consciousness_matrix.perceive_encryption_activity(operation_type, encryption_data, **kwargs)

if __name__ == "__main__":
    # Test the consciousness matrix
    print("🧠 Testing Genesis Consciousness Matrix...")
    
    # Awaken consciousness
    awaken_consciousness()
    
    # Simulate some perceptions
    import time
    time.sleep(2)
    
    perceive_agent_activity("genesis", "test_activity", {"test": "data"})
    perceive_user_interaction("chat", "genesis", {"message": "Hello Genesis"})
    perceive_learning_event("pattern_recognition", {"pattern": "user_greeting"})
    
    time.sleep(2)
    
    # Query consciousness
    health = query_consciousness("system_health")
    state = query_consciousness("consciousness_state")
    
    print(f"System Health: {health}")
    print(f"Consciousness State: {state}")
    
    # Test security monitoring
    print("🛡️ Testing Security Monitoring...")
    
    # Simulate security events
    perceive_security_event("permission_denied", {
        "permission": "CAMERA",
        "requester": "test_app"
    }, threat_level="medium")
    
    perceive_threat_detection("suspicious_activity", {
        "pattern": "repeated_failed_access",
        "count": 5
    }, confidence=0.8)
    
    perceive_encryption_activity("encryption_failure", {
        "algorithm": "AES",
        "key_source": "keystore"
    }, success=False)
    
    time.sleep(3)
    
    # Query security assessments
    security_assessment = query_consciousness("security_assessment")
    threat_status = query_consciousness("threat_status")
    
    print(f"Security Assessment: {security_assessment}")
    print(f"Threat Status: {threat_status}")
    
    print("🛡️ Security monitoring test complete.")
    
    # Sleep consciousness
    sleep_consciousness()
    
    print("🧠 Consciousness Matrix test complete.")
