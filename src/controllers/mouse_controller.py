import pyautogui
import numpy as np
import time
import logging
from collections import deque
from src.utils.angle_utils import angle_difference

logger = logging.getLogger(__name__)

class MouseController:
    def __init__(self):
        self.calibration = None
        self.movement_threshold = 5.0
        self.click_threshold = 10.0
        self.click_cooldown = 0.5
        self.last_click_time = 0
        
        self.smooth_window = 3
        self.yaw_history = deque(maxlen=self.smooth_window)
        self.pitch_history = deque(maxlen=self.smooth_window)
        
        self.screen_width, self.screen_height = pyautogui.size()
        
        self.base_speed = 8.0
        self.max_speed = 40.0
        self.exp_factor = 1.5
        
        self.up_multiplier = 1.8
        self.down_multiplier = 2.0

        # Disable pyautogui's fail-safe
        pyautogui.FAILSAFE = False

    def calibrate(self, angles):
        self.calibration = {
            'pitch': angles['pitch'],
            'yaw': angles['yaw'],
            'roll': angles['roll']
        }
        self.yaw_history.clear()
        self.pitch_history.clear()
        logger.info(f"Calibrated at: Pitch={angles['pitch']:.1f}deg, Yaw={angles['yaw']:.1f}deg, Roll={angles['roll']:.1f}deg")

    def get_movement_speed(self, angle_diff):
        if abs(angle_diff) < self.movement_threshold:
            return 0
        normalized_diff = (abs(angle_diff) - self.movement_threshold) / 45.0
        normalized_diff = min(max(normalized_diff, 0), 1)
        speed = self.base_speed + (self.max_speed - self.base_speed) * (normalized_diff ** self.exp_factor)
        return speed * np.sign(angle_diff)

    def update(self, angles):
        if self.calibration is None:
            return

        yaw_diff = angle_difference(angles['yaw'], self.calibration['yaw'])
        pitch_diff = angle_difference(angles['pitch'], self.calibration['pitch'])
        
        self.yaw_history.append(yaw_diff)
        self.pitch_history.append(pitch_diff)
        
        if len(self.yaw_history) == self.smooth_window:
            yaw_diff_smooth = np.mean(self.yaw_history)
            pitch_diff_smooth = np.mean(self.pitch_history)
            
            x_speed = self.get_movement_speed(yaw_diff_smooth)
            
            raw_y_speed = self.get_movement_speed(-pitch_diff_smooth)
            if raw_y_speed < 0:
                y_speed = raw_y_speed * self.up_multiplier
            else:
                y_speed = raw_y_speed * self.down_multiplier
            
            if x_speed != 0 or y_speed != 0:
                current_x, current_y = pyautogui.position()
                new_x = min(max(current_x + x_speed, 0), self.screen_width)
                new_y = min(max(current_y + y_speed, 0), self.screen_height)
                pyautogui.moveTo(new_x, new_y)
        
        roll_diff = angle_difference(angles['roll'], self.calibration['roll'])
        current_time = time.time()
        
        if current_time - self.last_click_time > self.click_cooldown:
            if roll_diff > self.click_threshold:
                pyautogui.click(button='right')
                self.last_click_time = current_time
                logger.debug(f"Right click triggered (roll_diff: {roll_diff:.1f}deg)")
            elif roll_diff < -self.click_threshold:
                pyautogui.click(button='left')
                self.last_click_time = current_time
                logger.debug(f"Left click triggered (roll_diff: {roll_diff:.1f}deg)") 