import math
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np

@dataclass
class SmartLightingEnvironmentConfig:
    """Configuration parameters for lighting physics and reward balancing."""
    time_step_hours: float = 0.5
    maximum_time_steps: float = 48

    occupied_target_illuminance: float = 400.0
    unoccupied_target_illuminance: float = 80.0

    minimum_colour_temperature: float = 2700.0
    maximum_colour_temperature: float = 6500.0

    led_illuminance_conversion_factor: float = 5.0
    led_power_conversion_factor: float = 0.02
    maximum_daylight_intensity: float = 350.0

    # Objective weights for multi-factor reward calculation
    illuminance_weight: float = 0.40
    power_consumption_weight: float = 0.40
    occupancy_awareness_weight: float = 0.10
    colour_temperature_weight: float = 0.05
    smoothness_weight: float = 0.05


class SmartLightingEnv(gym.Env):
    """
    Gymnasium environment for optimizing indoor lighting comfort and energy efficiency.
    The agent controls LED brightness and CCT based on occupancy and daylight.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: SmartLightingEnvironmentConfig | None = None):
        super().__init__()
        self.config = config or SmartLightingEnvironmentConfig()

        # Action: [Normalized Brightness, Normalized Color Temperature]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array( [1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [Total Lux, Daylight Lux, Occupancy, CCT, Time of Day]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        # Environmental and system state tracking
        self.current_time_step = 0
        self.time_of_day = 8.0
        self.daylight_intensity = 0.0
        self.occupancy_status = 0.0
        self.led_brightness_level = 0.0
        self.correlated_colour_temperature = 3000.0
        self.previous_led_brightness_level = 0.0
        self.previous_correlated_colour_temperature = 3000.0
        self.total_illuminance = 0.0
        self.target_illuminance = self.config.occupied_target_illuminance
        self.power_consumption = 0.0

    def reset(self, seed=None, options=None):
        """Reset environment to the start of a new 24-hour cycle."""
        super().reset(seed=seed)
        self.current_time_step = 0
        self.time_of_day = 0.0

        # Update conditions based on initial time
        self.daylight_intensity = self._calculate_daylight_intensity(self.time_of_day)
        self.occupancy_status = self._calculate_occupancy_status(self.time_of_day)
        self.target_illuminance = self._get_target_illuminance()

        # Initialize hardware state
        self.led_brightness_level = self._calculate_recommended_led_brightness_level()
        self.correlated_colour_temperature = self._get_target_colour_temperature(self.time_of_day)
        self.previous_led_brightness_level = self.led_brightness_level
        self.previous_correlated_colour_temperature = self.correlated_colour_temperature

        self.total_illuminance = self._calculate_total_illuminance()
        self.power_consumption = self._calculate_power_consumption()

        return self._get_observation(), self._build_info(0.0)

    def step(self, action):
        """Apply agent actions, update lighting physics, and transition time."""
        self.previous_led_brightness_level = self.led_brightness_level
        self.previous_correlated_colour_temperature = self.correlated_colour_temperature

        # Denormalize actions to physical control signals
        normalized_brightness_action = float(np.clip(action[0], -1.0, 1.0))
        normalized_colour_temperature_action = float(np.clip(action[1], -1.0, 1.0))

        self.led_brightness_level = self._map_normalized_brightness_to_real_value(
            normalized_brightness_action
        )
        self.correlated_colour_temperature = (
            self._map_normalized_colour_temperature_to_real_value(
                normalized_colour_temperature_action
            )
        )

        # Calculate current step performance
        self.total_illuminance = self._calculate_total_illuminance()
        self.power_consumption = self._calculate_power_consumption()
        reward = self._calculate_reward()
        step_info = self._build_info(reward)

        # State transition logic
        self.current_time_step += 1
        truncated = self.current_time_step >= self.config.maximum_time_steps
        terminated = False

        if not (terminated or truncated):
            # Advance environment clock and update external variables
            self.time_of_day = (self.time_of_day + self.config.time_step_hours) % 24.0
            self.daylight_intensity = self._calculate_daylight_intensity(self.time_of_day)
            self.occupancy_status = self._calculate_occupancy_status(self.time_of_day)
            self.target_illuminance = self._get_target_illuminance()
            
            # Refresh system physics for the next observation
            self.total_illuminance = self._calculate_total_illuminance()
            self.power_consumption = self._calculate_power_consumption()

        return self._get_observation(), reward, terminated, truncated, step_info

    def render(self):
        """Print current system status to console."""
        print(
            f"Time of day: {self.time_of_day:.1f} h | "
            f"Occupancy status: {int(self.occupancy_status)} | "
            f"Daylight intensity: {self.daylight_intensity:.1f} lux | "
            f"LED brightness level: {self.led_brightness_level:.1f}% | "
            f"Correlated colour temperature: {self.correlated_colour_temperature:.0f} K | "
            f"Total illuminance: {self.total_illuminance:.1f} lux | "
            f"Target illuminance: {self.target_illuminance:.1f} lux | "
            f"Power consumption: {self.power_consumption:.3f}"
        )

    def _get_observation(self):
        """Normalize internal state variables to [0, 1] range for RL input."""
        return np.array(
            [
                np.clip(self.total_illuminance / 1000.0, 0.0, 1.0),
                np.clip(self.daylight_intensity / 600.0, 0.0, 1.0),
                self.occupancy_status,
                np.clip(self.correlated_colour_temperature / 6500.0, 0.0, 1.0),
                np.clip(self.time_of_day / 24.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    """Construct diagnostic metrics used for evaluation and analysis."""
    def _build_info(self, reward: float):
        """Construct detailed diagnostic dictionary for logging."""
        illuminance_error = abs(self.target_illuminance - self.total_illuminance)

        comfort_score = self._calculate_lighting_comfort_score()
        energy_efficiency_score = self._calculate_energy_efficiency_score()
        colour_temperature_score = self._calculate_colour_temperature_score()
        overall_score = (
            0.55 * comfort_score
            + 0.25 * energy_efficiency_score
            + 0.20 * colour_temperature_score
        )

        return {
            "time_of_day": self.time_of_day,
            "daylight_intensity": self.daylight_intensity,
            "occupancy_status": self.occupancy_status,
            "led_brightness_level": self.led_brightness_level,
            "correlated_colour_temperature": self.correlated_colour_temperature,
            "total_illuminance": self.total_illuminance,
            "target_illuminance": self.target_illuminance,
            "illuminance_error": illuminance_error,
            "power_consumption": self.power_consumption,
            "comfort_score": comfort_score,
            "energy_efficiency_score": energy_efficiency_score,
            "colour_temperature_score": colour_temperature_score,
            "overall_score": overall_score,
            "reward": reward,
            "recommended_led_brightness_level": self._calculate_recommended_led_brightness_level(),
            "recommended_colour_temperature": self._get_target_colour_temperature(self.time_of_day),
        }

    def _map_normalized_brightness_to_real_value(self, normalized_brightness_action: float):
        """Maps action range [-1, 1] to hardware brightness [0, 100]."""
        return float((normalized_brightness_action + 1.0) * 50.0)

    def _map_normalized_colour_temperature_to_real_value(self, normalized_colour_temperature_action: float):
        """Maps action range [-1, 1] to Kelvin range defined in config."""
        colour_temperature_range = (
            self.config.maximum_colour_temperature
            - self.config.minimum_colour_temperature
        )
        return float(
            self.config.minimum_colour_temperature
            + ((normalized_colour_temperature_action + 1.0) / 2.0)
            * colour_temperature_range
        )

    def _get_target_illuminance(self):
        """Retrieve lux setpoint based on current occupancy."""
        if self.occupancy_status == 1.0:
            return self.config.occupied_target_illuminance
        return self.config.unoccupied_target_illuminance

    def _get_target_colour_temperature(self, current_time_of_day: float):
        """Schedule-based CCT targets for circadian rhythm alignment."""
        if 8.0 <= current_time_of_day < 12.0:
            return 5500.0
        if 12.0 <= current_time_of_day < 17.0:
            return 5000.0
        if 17.0 <= current_time_of_day < 21.0:
            return 3500.0
        return 3000.0

    def _calculate_daylight_intensity(self, current_time_of_day: float):
        """Sine-wave approximation of daylight lux over a 12-hour sun cycle."""
        if 6.0 <= current_time_of_day <= 18.0:
            daylight_phase = (current_time_of_day - 6.0) / 12.0
            return float(
                self.config.maximum_daylight_intensity
                * math.sin(math.pi * daylight_phase)
            )
        return 0.0

    def _calculate_occupancy_status(self, current_time_of_day: float):
        """Simplified binary office occupancy schedule."""
        if 8.0 <= current_time_of_day < 12.0:
            return 1.0
        if 13.0 <= current_time_of_day < 18.0:
            return 1.0
        return 0.0

    def _calculate_led_contributed_illuminance(self):
        """Physics: Convert brightness percentage to lux contribution."""
        return self.config.led_illuminance_conversion_factor * self.led_brightness_level

    def _calculate_total_illuminance(self):
        """Physics: Superposition of artificial and natural light."""
        return self._calculate_led_contributed_illuminance() + self.daylight_intensity

    def _calculate_power_consumption(self):
        """Estimate relative power usage based on LED brightness."""
        return self.config.led_power_conversion_factor * self.led_brightness_level

    def _calculate_recommended_led_brightness_level(self):
        """Calculate mathematically optimal brightness to meet target lux."""
        required_led_illuminance = max(
            0.0,
            self.target_illuminance - self.daylight_intensity,
        )
        recommended_led_brightness_level = (
            required_led_illuminance / self.config.led_illuminance_conversion_factor
        )
        return float(np.clip(recommended_led_brightness_level, 0.0, 100.0))

    def _calculate_lighting_comfort_score(self):
        """Heuristic score (0-100) for user visual comfort."""
        illuminance_error = abs(self.target_illuminance - self.total_illuminance)
        if self.occupancy_status == 1.0:
            score = 100.0 - 0.6 * illuminance_error
        else:
            excess_light = max(
                0.0,
                self.total_illuminance - self.config.unoccupied_target_illuminance,
            )
            score = 100.0 - 0.8 * excess_light
        return float(np.clip(score, 0.0, 100.0))

    def _calculate_energy_efficiency_score(self):
        """Heuristic score (0-100) for power conservation."""
        recommended_led_brightness_level = self._calculate_recommended_led_brightness_level()
        brightness_difference = abs(self.led_brightness_level - recommended_led_brightness_level)
        if self.occupancy_status == 0.0:
            score = 100.0 - 1.2 * self.led_brightness_level
        else:
            score = 100.0 - 0.8 * brightness_difference
        return float(np.clip(score, 0.0, 100.0))

    def _calculate_colour_temperature_score(self):
        """Heuristic score (0-100) for circadian CCT alignment."""
        target_colour_temperature = self._get_target_colour_temperature(self.time_of_day)
        colour_temperature_difference = abs(
            self.correlated_colour_temperature - target_colour_temperature
        )
        score = 100.0 - 0.08 * colour_temperature_difference
        return float(np.clip(score, 0.0, 100.0))

    def _calculate_reward(self):
        """
        Multi-objective reward function that balances:
        - lighting comfort (illuminance accuracy)
        - energy efficiency (power consumption)
        - occupancy awareness
        - smooth control transitions
        """
        recommended_led_brightness_level = self._calculate_recommended_led_brightness_level()
        target_colour_temperature = self._get_target_colour_temperature(self.time_of_day)

        illuminance_difference = abs(self.target_illuminance - self.total_illuminance)
        brightness_difference = abs(self.led_brightness_level - recommended_led_brightness_level)
        colour_temperature_difference = abs(
            self.correlated_colour_temperature - target_colour_temperature
        )

        # Normalize penalties to scale between [0, 1]
        normalised_illuminance_penalty = min(illuminance_difference / 400.0, 1.0)
        normalised_power_penalty = min(self.power_consumption / 2.0, 1.0)
        normalised_colour_temperature_penalty = min(colour_temperature_difference / 3000.0, 1.0)

        occupancy_awareness_penalty = 0.0
        if self.occupancy_status == 0.0 and self.led_brightness_level > 10.0:
            occupancy_awareness_penalty = min((self.led_brightness_level - 10.0) / 90.0, 1.0)

        smoothness_penalty = min(
            (
                abs(self.led_brightness_level - self.previous_led_brightness_level)
                + abs(self.correlated_colour_temperature - self.previous_correlated_colour_temperature) / 100.0
            ) / 80.0, 1.0,
        )

        reward = 1.0
        reward -= self.config.illuminance_weight * normalised_illuminance_penalty
        reward -= self.config.power_consumption_weight * normalised_power_penalty
        reward -= self.config.occupancy_awareness_weight * occupancy_awareness_penalty
        reward -= self.config.colour_temperature_weight * normalised_colour_temperature_penalty
        reward -= self.config.smoothness_weight * smoothness_penalty

        # Task-specific bonuses and hard penalties
        if self.occupancy_status == 1.0 and illuminance_difference <= 20.0:
            reward += 0.30
        elif self.occupancy_status == 1.0 and illuminance_difference > 120.0 and self.led_brightness_level < 10.0:
            reward -= 0.15

        if self.occupancy_status == 0.0 and self.led_brightness_level <= 10.0:
            reward += 0.08

        if brightness_difference <= 5.0:
            reward += 0.08

        return float(np.clip(reward, -1.0, 1.5))