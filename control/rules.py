"""
Traffic rules engine with 7-level priority.

Decides what the car should do based on perception.
"""

from dataclasses import dataclass
from typing import Optional
from perception.runtime import PerceptionOutput
from control.runtime import ControlDecision, DecisionReason, ControlConfig


@dataclass(frozen=True)
class TrafficRulesConfig:
    """Configuration for traffic rules."""
    # Inherit from ControlConfig for consistency
    pass


class TrafficRulesEngine:
    """Decision engine with 7-level priority.
    
    Priority order (checked in sequence):
    1. Emergency (lane lost + hazard) → immediate brake
    2. Red light → brake
    3. Pedestrian detected → brake or avoid
    4. Obstacle detected → brake or avoid
    5. Hazard detected → brake
    6. Lane lost → brake
    7. Clear path → cruise
    """
    
    def __init__(self, config: Optional[ControlConfig] = None):
        self.config = config or ControlConfig()
    
    def decide(self, perception: PerceptionOutput) -> ControlDecision:
        """Make decision based on perception output.
        
        Returns ControlDecision with reason, speed, steering, brake flag.
        """
        
        # Level 1: Emergency (lane lost + hazard)
        result = self._check_emergency(perception)
        if result:
            return result
        
        # Level 2: Red light
        result = self._check_red_light(perception)
        if result:
            return result
        
        # Level 3: Pedestrians
        result = self._check_pedestrians(perception)
        if result:
            return result
        
        # Level 4: Obstacles
        result = self._check_obstacles(perception)
        if result:
            return result
        
        # Level 5: Hazards
        result = self._check_hazards(perception)
        if result:
            return result
        
        # Level 6: Lane lost
        result = self._check_lane_lost(perception)
        if result:
            return result
        
        # Level 7: Clear path
        return self._check_clear_path(perception)
    
    def _check_emergency(self, perception: PerceptionOutput) -> Optional[ControlDecision]:
        """Check for emergency: lane lost + hazard."""
        lane_lost = perception.laneseg is None or perception.laneseg.conf < self.config.min_laneseg_conf
        hazard_present = perception.hazard is not None and perception.hazard.conf >= self.config.min_hazard_conf
        
        if lane_lost and hazard_present:
            return ControlDecision(
                reason=DecisionReason.EMERGENCY,
                target_speed=0.0,
                target_steer=0.0,
                should_brake=True,
                confidence=0.95,
            )
        return None
    
    def _check_red_light(self, perception: PerceptionOutput) -> Optional[ControlDecision]:
        """Check for red light."""
        if perception.traffic_light is None:
            return None
        
        if perception.traffic_light.conf < self.config.min_traffic_light_conf:
            return None
        
        if perception.traffic_light.state == "red":
            return ControlDecision(
                reason=DecisionReason.RED_LIGHT,
                target_speed=0.0,
                target_steer=0.0,
                should_brake=True,
                confidence=perception.traffic_light.conf,
            )
        return None
    
    def _check_pedestrians(self, perception: PerceptionOutput) -> Optional[ControlDecision]:
        """Check for pedestrians and decide brake/avoid."""
        pedestrians = [
            obj for obj in perception.objects
            if obj.cls == "pedestrian" and obj.conf >= self.config.min_object_conf
        ]
        
        if not pedestrians:
            return None
        
        # Find closest pedestrian
        closest = min(pedestrians, key=lambda p: self._bbox_distance(p.bbox))
        distance = self._bbox_distance(closest.bbox)
        
        # Too close: brake
        if distance < self.config.dangerous_dist_threshold:
            return ControlDecision(
                reason=DecisionReason.PEDESTRIAN_DETECTED,
                target_speed=0.0,
                target_steer=0.0,
                should_brake=True,
                confidence=closest.conf,
            )
        
        # Need caution: slow down and maybe avoid
        if distance < self.config.caution_dist_threshold:
            steer = self._compute_avoidance_steer(closest.bbox, perception)
            return ControlDecision(
                reason=DecisionReason.PEDESTRIAN_DETECTED,
                target_speed=self.config.safe_speed,
                target_steer=steer,
                should_brake=False,
                confidence=closest.conf * 0.8,
            )
        
        return None
    
    def _check_obstacles(self, perception: PerceptionOutput) -> Optional[ControlDecision]:
        """Check for obstacles and decide brake/avoid."""
        obstacle_classes = {"obstacle", "roadblock", "bollard", "bicycle", "car", "scooter"}
        obstacles = [
            obj for obj in perception.objects
            if obj.cls in obstacle_classes
            and obj.conf >= self.config.min_object_conf
        ]
        
        if not obstacles:
            return None
        
        # Find closest obstacle
        closest = min(obstacles, key=lambda o: self._bbox_distance(o.bbox))
        distance = self._bbox_distance(closest.bbox)
        
        # Too close: brake
        if distance < self.config.dangerous_dist_threshold:
            return ControlDecision(
                reason=DecisionReason.OBSTACLE_DETECTED,
                target_speed=0.0,
                target_steer=0.0,
                should_brake=True,
                confidence=closest.conf,
            )
        
        # Need caution: slow down and avoid
        if distance < self.config.caution_dist_threshold:
            steer = self._compute_avoidance_steer(closest.bbox, perception)
            return ControlDecision(
                reason=DecisionReason.OBSTACLE_DETECTED,
                target_speed=self.config.safe_speed,
                target_steer=steer,
                should_brake=False,
                confidence=closest.conf * 0.8,
            )
        
        return None
    
    def _check_hazards(self, perception: PerceptionOutput) -> Optional[ControlDecision]:
        """Check for road hazards (potholes, etc.)."""
        if perception.hazard is None:
            return None
        
        if perception.hazard.conf < self.config.min_hazard_conf:
            return None
        
        return ControlDecision(
            reason=DecisionReason.HAZARD_DETECTED,
            target_speed=0.0,
            target_steer=0.0,
            should_brake=True,
            confidence=perception.hazard.conf,
        )
    
    def _check_lane_lost(self, perception: PerceptionOutput) -> Optional[ControlDecision]:
        """Check if lane/drivable area is lost."""
        if perception.laneseg is None or perception.laneseg.conf < self.config.min_laneseg_conf:
            return ControlDecision(
                reason=DecisionReason.LANE_LOST,
                target_speed=0.0,
                target_steer=0.0,
                should_brake=True,
                confidence=0.5,
            )
        return None
    
    def _check_clear_path(self, perception: PerceptionOutput) -> ControlDecision:
        """No obstacles: proceed normally."""
        # Check traffic light state
        if perception.traffic_light is not None and perception.traffic_light.conf >= self.config.min_traffic_light_conf:
            if perception.traffic_light.state in ["yellow", "flashing"]:
                # Caution: slow down
                return ControlDecision(
                    reason=DecisionReason.CLEAR_PATH,
                    target_speed=self.config.safe_speed,
                    target_steer=0.0,
                    should_brake=False,
                    confidence=0.8,
                )
        
        # Green light or no signal: cruise
        return ControlDecision(
            reason=DecisionReason.CLEAR_PATH,
            target_speed=self.config.max_speed,  # Changed from cruise_speed to max_speed
            target_steer=0.0,
            should_brake=False,
            confidence=1.0,
        )
    
    @staticmethod
    def _bbox_distance(bbox: list) -> float:
        """Compute distance as proximity to image center.
        
        Lower value = closer to center = more dangerous.
        
        Uses vertical position (y) as proxy for distance.
        """
        # bbox: [x1, y1, x2, y2]
        y_center = (bbox[1] + bbox[3]) / 2
        # Assume 480px height, closer to bottom = closer to camera
        # Distance = 480 - y_center (inverted)
        return 480 - y_center
    
    @staticmethod
    def _compute_avoidance_steer(bbox: list, perception: PerceptionOutput) -> float:
        """Compute steering angle to avoid obstacle.
        
        bbox: [x1, y1, x2, y2] in pixels
        
        Simple heuristic: if object is on left, steer right (positive).
        If on right, steer left (negative).
        """
        # Image center (assume 640px width)
        img_center_x = 320
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        
        # Offset from center (-320 to +320)
        offset = bbox_center_x - img_center_x
        
        # Convert to steering angle (±30°)
        # -320 (far left) → +30° (steer right)
        # +320 (far right) → -30° (steer left)
        steer = -offset / 320 * 30.0
        
        # Clamp
        steer = max(-30.0, min(30.0, steer))
        
        return steer
