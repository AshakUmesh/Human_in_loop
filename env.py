''' 
This environment describe a fixed scene (area) to conduct end-to-end lateral control tasks
for the autonomous ego vehicle. (This environment is relative simple and is only for training)
'''

'''
CARLA 0.9.15 compatible environment for Human-in-the-Loop DRL.
Changes from original (CARLA 0.9.7):
  - Removed joystick/steering wheel; replaced with mouse + keyboard input
  - set_velocity() replaced with set_target_velocity()
  - Synchronous mode properly managed (tick required)
  - sys.path.append replaced with carla pip package import
  - All actor destruction done safely with is_alive checks
  - Human intervention: SPACEBAR to toggle takeover
    W/S = throttle/brake, A/D or mouse X = steering

NEW FEATURES ADDED:
  - Camera wait on restart (prevents Fatal Python error)
  - Collision warning: screen flashes RED + terminal warning
    when car is about to hit something
  - Episode data saved automatically every episode to episode_data/
  - CARLA stays open between episodes
  - World settings applied only once not every episode
'''

import pygame
import weakref
import collections
import numpy as np
import math
import cv2
import re
import sys
import time
import os

import carla
from carla import ColorConverter as cc

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

from utils import get_path
path_generator = get_path()

velocity_target_ego = 5
x_bench = 335.0
y_bench = 200.0
WIDTH, HEIGHT = 80, 45

# Danger thresholds - tune these to control warning sensitivity
DANGER_DISTANCE_FRONT  = 0.5
DANGER_DISTANCE_SIDE   = 0.4
DANGER_DISTANCE_CORNER = 0.3


# ──────────────────────────────────────────────────────────────
#  Mouse + Keyboard Human Input Handler
# ──────────────────────────────────────────────────────────────
class HumanInputHandler:
    """
    Controls:
      1. Single LEFT click on pygame window to focus it
      2. Press SPACEBAR to toggle human takeover ON/OFF
      3. When takeover is ON:
         W / Up Arrow    -> Accelerate
         S / Down Arrow  -> Brake
         A / Left Arrow  -> Steer Left
         D / Right Arrow -> Steer Right
         Mouse Left/Right -> Analog Steering
      4. Press SPACEBAR again to give control back to AI
    """
    MOUSE_SENSITIVITY = 0.004
    KEY_STEER_STEP    = 0.1
    STEER_DECAY       = 0.82

    def __init__(self):
        self._steer        = 0.0
        self._throttle     = 0.0
        self._brake        = 0.0
        self._takeover     = False
        self._prev_mouse_x = None
        

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._takeover = not self._takeover
                if not self._takeover:
                    self._steer    = 0.0
                    self._throttle = 0.0
                    self._brake    = 0.0
                status = ("HUMAN TAKEOVER ON"
                          if self._takeover else "AI CONTROL ON")
                print(f">>> {status} <<<")

        elif event.type == pygame.MOUSEMOTION and self._takeover:
            dx, _ = event.rel
            self._steer += dx * self.MOUSE_SENSITIVITY
            self._steer = float(np.clip(self._steer, -1.0, 1.0))

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pass

    def update(self):
        if not self._takeover:
            self._steer    *= self.STEER_DECAY
            self._throttle  = 0.0
            self._brake     = 0.0
            return

        keys = pygame.key.get_pressed()

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self._steer -= self.KEY_STEER_STEP
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self._steer += self.KEY_STEER_STEP
        else:
            self._steer *= 0.85

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self._throttle = min(self._throttle + 0.05, 1.0)
            self._brake    = 0.0
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self._brake    = min(self._brake + 0.05, 1.0)
            self._throttle = 0.0
        else:
            self._throttle *= 0.9
            self._brake    *= 0.9

        self._steer = float(np.clip(self._steer, -1.0, 1.0))

    @property
    def steer(self):
        return self._steer

    @property
    def throttle(self):
        return self._throttle

    @property
    def brake(self):
        return self._brake

    @property
    def is_takeover(self):
        return self._takeover

    def get_steer_history_value(self):
        return self._steer


# ──────────────────────────────────────────────────────────────
#  CARLA Scenario Environment
# ──────────────────────────────────────────────────────────────
class scenario(object):
    def __init__(self, random_spawn=False, pedestrian=False,
                 no_render=False, frame=25):

        self.observation_size_width  = WIDTH
        self.observation_size_height = HEIGHT
        self.observation_size        = WIDTH * HEIGHT
        self.action_size             = 1

        self.pedestrian   = pedestrian
        self.random_spawn = random_spawn
        self.no_render    = no_render
        self.frame        = frame

        self.ego_vehicle      = None
        self.obs1             = None
        self.obs2             = None
        self.obs3             = None
        self.collision_sensor = None
        self.seman_camera     = None
        self.viz_camera       = None

        self.surface       = None
        self.camera_output = np.zeros([720, 1280, 3], dtype=np.uint8)
        self.recording     = False

        # Warning state for collision alert
        self.danger_warning  = False
        self.warning_counter = 0

        # Episode tracking
        self.episode_count   = 0
        self.episode_rewards = []
        self.episode_steps   = []

        self.Attachment = carla.AttachmentType

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(120.0)

        self.world = self.client.load_world('Town01')

        pygame.init()
        pygame.font.init()
        self.display    = pygame.display.set_mode(
            (1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.infoObject = pygame.display.Info()
        pygame.display.set_caption(
            'HITL-DRL | SPACE=takeover | WASD=drive | '
            'Screen flashes RED on danger')

        self.human_input = HumanInputHandler()

        # World settings flag - only apply once
        self._settings_applied = False

        # Create episode data folder
        os.makedirs('episode_data', exist_ok=True)

    # ──────────────────────────────────────────────────────────
    def restart(self):

        # Apply world settings only on very first episode
        if not self._settings_applied:
            self.original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode    = True
            settings.fixed_delta_seconds = 1.0 / self.frame
            self.world.apply_settings(settings)
            tm = self.client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            self._settings_applied = True

        # Reset camera and surface for new episode
        self.camera_output = np.zeros([720, 1280, 3], dtype=np.uint8)
        self.surface       = None

        # Reset warning
        self.danger_warning  = False
        self.warning_counter = 0

        # Goal marker
        goal_loc = carla.Location(x=x_bench, y=y_bench + 500.0, z=1)
        self.world.debug.draw_point(
            goal_loc, size=0.1,
            color=carla.Color(r=255, g=0, b=0),
            life_time=1000)

        # Weather
        self._weather_presets = find_weather_presets()
        self._weather_index   = 0
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])

        # Reset histories
        self.steer_history     = []
        self.intervene_history = []
        self.intervention      = False
        self.step_reward_list  = []

        # Spawn surrounding vehicles
        self.bp_obs1, self.spawn_point_obs1 = \
            self._produce_vehicle_blueprint(1, 335.0 + 3.5, 100.0)
        self.obs1 = self.world.spawn_actor(
            self.bp_obs1, self.spawn_point_obs1)

        self.bp_obs2, self.spawn_point_obs2 = \
            self._produce_vehicle_blueprint(1, 335.0, 200.0 + 25.0)
        self.obs2 = self.world.spawn_actor(
            self.bp_obs2, self.spawn_point_obs2)

        self.bp_obs3, self.spawn_point_obs3 = \
            self._produce_vehicle_blueprint(
                1, 335.0 + 3.5, 200.0 + 50.0)
        self.obs3 = self.world.spawn_actor(
            self.bp_obs3, self.spawn_point_obs3)

        if self.pedestrian:
            self.bp_walker1, self.spawn_point_walker1 = \
                self._produce_walker_blueprint(
                    338.0, 200 + np.random.randint(10, 15))
            self.bp_walker2, self.spawn_point_walker2 = \
                self._produce_walker_blueprint(
                    np.random.randint(3310, 3350) / 10, 235)
            self.walker1 = self.world.spawn_actor(
                self.bp_walker1, self.spawn_point_walker1)
            self.walker2 = self.world.spawn_actor(
                self.bp_walker2, self.spawn_point_walker2)
            self.walker1.apply_control(carla.WalkerControl(speed=0.1))
            self.walker2.apply_control(carla.WalkerControl(speed=0.1))

        # Spawn ego vehicle
        if self.random_spawn:
            y_spawn_random = np.random.randint(200, 240)
            x_spawn_random = path_generator(y_spawn_random) + \
                0.1 * (np.random.rand() - 0.5)
            self.bp_ego, self.spawn_point_ego = \
                self._produce_vehicle_blueprint(
                    1, x_spawn_random, y_spawn_random)
        else:
            self.bp_ego, self.spawn_point_ego = \
                self._produce_vehicle_blueprint(1, x_bench, y_bench)

        self.ego_vehicle = self.world.spawn_actor(
            self.bp_ego, self.spawn_point_ego)

        initial_velocity = carla.Vector3D(0, velocity_target_ego, 0)
        self.ego_vehicle.set_target_velocity(initial_velocity)

        self.control = carla.VehicleControl()

        # Collision sensor
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find(
            'sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            bp_collision, carla.Transform(),
            attach_to=self.ego_vehicle)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: scenario._on_collision(weak_self, event))

        # Camera transforms
        self.camera_transforms = [
            (carla.Transform(
                carla.Location(x=-2, z=5),
                carla.Rotation(pitch=30.0)),
             self.Attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-2, z=5),
                carla.Rotation(pitch=30.0)),
             self.Attachment.SpringArm),
        ]
        self.camera_transform_index = 1

        self.cameras = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.semantic_segmentation',
             cc.CityScapesPalette,
             'Camera Semantic Segmentation', {}],
        ]

        # RGB camera
        bp_viz_camera = self.world.get_blueprint_library().find(
            'sensor.camera.rgb')
        bp_viz_camera.set_attribute('image_size_x', '1280')
        bp_viz_camera.set_attribute('image_size_y', '720')
        bp_viz_camera.set_attribute('sensor_tick', '0.02')
        self.cameras[0].append(bp_viz_camera)

        # Semantic camera
        bp_seman_camera = self.world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
        bp_seman_camera.set_attribute('image_size_x', '1280')
        bp_seman_camera.set_attribute('image_size_y', '720')
        bp_seman_camera.set_attribute('sensor_tick', '0.04')
        self.cameras[1].append(bp_seman_camera)

        # Safely destroy existing cameras
        for cam in [self.seman_camera, self.viz_camera]:
            try:
                if cam is not None and cam.is_alive:
                    cam.stop()
                    cam.destroy()
            except Exception:
                pass
        self.seman_camera = None
        self.viz_camera   = None

        cam_transform = self.camera_transforms[
            self.camera_transform_index]
        self.viz_camera = self.world.spawn_actor(
            self.cameras[0][-1],
            cam_transform[0],
            attach_to=self.ego_vehicle,
            attachment_type=cam_transform[1])

        seman_transform = self.camera_transforms[
            self.camera_transform_index - 1]
        self.seman_camera = self.world.spawn_actor(
            self.cameras[1][-1],
            seman_transform[0],
            attach_to=self.ego_vehicle,
            attachment_type=seman_transform[1])

        weak_self = weakref.ref(self)
        self.seman_camera.listen(
            lambda image: scenario._parse_seman_image(
                weak_self, image))
        self.viz_camera.listen(
            lambda image: scenario._parse_image(weak_self, image))

        self.count = 0

        # Tick multiple times to initialise sensors properly
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)

        # Wait until camera has actual data (CRITICAL - prevents crash)
        wait_count = 0
        while np.sum(self.camera_output) == 0 and wait_count < 30:
            self.world.tick()
            time.sleep(0.1)
            wait_count += 1
            if wait_count % 10 == 0:
                print(f'Waiting for camera... ({wait_count}/30)')

        state, other_indicators = self.obtain_observation()

        print(f'\n--- Episode {self.episode_count + 1} starts '
              f'in 3 seconds ---')
        print('Click pygame window then SPACEBAR to take over')
        time.sleep(3)

        return state, other_indicators

    # ──────────────────────────────────────────────────────────
    def render(self, display):
        if self.surface is not None:
            m = pygame.transform.smoothscale(
                self.surface,
                [int(self.infoObject.current_w),
                 int(self.infoObject.current_h)])
            display.blit(m, (0, 0))

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _parse_seman_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[1][1])
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_output = array

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[0][1])
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(
            array.swapaxes(0, 1))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse   = event.normal_impulse
        intensity = math.sqrt(
            impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)

    # ──────────────────────────────────────────────────────────
    def get_collision_history(self):
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag

    # ──────────────────────────────────────────────────────────
    def _check_danger(self, other_indicators):
        """
        Checks if the car is dangerously close to anything.
        Shows visual warning on screen and prints to terminal.
        Human should press SPACEBAR when this triggers.
        """
        dis_to_front = other_indicators['state_front']
        dis_to_side  = min(other_indicators['state_left'],
                           other_indicators['state_right'])
        min_corner   = min(
            other_indicators['state_corner_11'],
            other_indicators['state_corner_12'],
            other_indicators['state_corner_21'],
            other_indicators['state_corner_22'],
            other_indicators['state_corner_31'],
            other_indicators['state_corner_32'])

        danger = False
        reason = ''

        if dis_to_front < DANGER_DISTANCE_FRONT:
            danger = True
            reason = f'FRONT OBSTACLE ({dis_to_front:.2f})'
        elif dis_to_side < DANGER_DISTANCE_SIDE:
            danger = True
            reason = f'ROAD EDGE ({dis_to_side:.2f})'
        elif min_corner < DANGER_DISTANCE_CORNER:
            danger = True
            reason = f'VEHICLE NEAR ({min_corner:.2f})'

        # Print warning only when danger first detected
        if danger and not self.danger_warning:
            print(f'\n*** DANGER: {reason} ***')
            print('    >>> PRESS SPACEBAR NOW TO TAKE OVER! <<<\n')

        self.danger_warning = danger
        return danger

    # ──────────────────────────────────────────────────────────
    def run_step(self, action):
        """
        Runs one simulation step.
        Returns: next_state, human_control, reward,
                 intervention_flag, done, physical_vars
        """

        self.render(self.display)
        self._draw_hud()
        pygame.display.flip()

        self.parse_events()
        self.human_input.update()

        human_control = None

        vx = self.ego_vehicle.get_velocity().x
        vy = self.ego_vehicle.get_velocity().y
        velocity_ego = math.sqrt(vx**2 + vy**2)

        if not self.human_input.is_takeover:
            steerCmd              = action / 2
            self.control.steer    = math.tan(1.1 * steerCmd)
            self.control.throttle = float(np.clip(
                velocity_target_ego - velocity_ego, 0, 1))
            self.control.brake    = 0.0
            self.intervention     = False
        else:
            steerCmd              = self.human_input.steer
            self.control.steer    = float(steerCmd)
            self.control.throttle = float(self.human_input.throttle)
            self.control.brake    = float(self.human_input.brake)
            human_control         = steerCmd
            self.intervention     = True

        self.control.hand_brake        = False
        self.control.reverse           = False
        self.control.manual_gear_shift = False

        self.intervene_history.append(
            self.human_input.get_steer_history_value())
        self.steer_history.append(steerCmd)

        self.ego_vehicle.apply_control(self.control)
        self.world.tick()

        next_states, other_indicators = self.obtain_observation()

        # Check danger and show warning
        self._check_danger(other_indicators)

        collision = self.get_collision_history()[1]
        finish    = (self.ego_vehicle.get_location().y
                     > y_bench + 55.0)
        beyond    = ((self.ego_vehicle.get_location().x
                      < x_bench - 1.2) or
                     (self.ego_vehicle.get_location().x
                      > x_bench + 4.8))
        done = bool(collision or finish or beyond)

        dis_to_front = other_indicators['state_front']
        dis_to_side  = min(other_indicators['state_left'],
                           other_indicators['state_right'])
        dis_to_obs11 = other_indicators['state_corner_11']
        dis_to_obs12 = other_indicators['state_corner_12']
        dis_to_obs21 = other_indicators['state_corner_21']
        dis_to_obs22 = other_indicators['state_corner_22']
        dis_to_obs31 = other_indicators['state_corner_31']
        dis_to_obs32 = other_indicators['state_corner_32']

        r1 = -1 * np.square(1 - dis_to_front)
        r2 = -2 * np.square(1 - dis_to_side)
        r3 = -(np.abs(1 - dis_to_obs11) + np.abs(1 - dis_to_obs12) +
               np.abs(1 - dis_to_obs21) + np.abs(1 - dis_to_obs22) +
               np.abs(1 - dis_to_obs31) + np.abs(1 - dis_to_obs32))
        r4 = finish * 10 - collision * 10 - beyond * 10
        r5 = -np.float32(
            abs(self.steer_history[-1] - steerCmd) > 0.1)
        r6 = -3 * abs(steerCmd)

        reward = float(np.clip(
            r1 + r2 + r3 + r4 + r5 + r6 + 0.2, -10, 10))

        self.step_reward_list.append(reward)
        self.count += 1

        vx_now = self.ego_vehicle.get_velocity().x
        vy_now = self.ego_vehicle.get_velocity().y
        yaw_rate = (math.atan(vx_now / vy_now)
                    if vy_now > 0 else 0)

        physical_variables = {
            'velocity_y':         vy_now,
            'velocity_x':         vx_now,
            'position_y':         self.ego_vehicle.get_location().y,
            'position_x':         self.ego_vehicle.get_location().x,
            'yaw_rate':           yaw_rate,
            'yaw':                self.ego_vehicle.get_transform(
                ).rotation.yaw,
            'pitch':              self.ego_vehicle.get_transform(
                ).rotation.pitch,
            'roll':               self.ego_vehicle.get_transform(
                ).rotation.roll,
            'angular_velocity_y': self.ego_vehicle.get_angular_velocity().y,
            'angular_velocity_x': self.ego_vehicle.get_angular_velocity().x,
        }

        if done:
            self._save_episode_data(finish, collision)
            self.post_process()

        return (next_states, human_control, reward,
                self.intervention, done, physical_variables)

    # ──────────────────────────────────────────────────────────
    def _save_episode_data(self, finish, collision):
        """Save episode result to file after every episode."""
        self.episode_count += 1
        steps      = self.count
        avg_reward = (np.mean(self.step_reward_list)
                      if self.step_reward_list else 0.0)
        result     = ('SUCCESS'   if finish
                      else 'COLLISION' if collision
                      else 'OUT_OF_ROAD')

        self.episode_rewards.append(avg_reward)
        self.episode_steps.append(steps)

        # Append to text log
        log_path = 'episode_data/episode_log.txt'
        with open(log_path, 'a') as f:
            f.write(f'Episode {self.episode_count:4d} | '
                    f'Steps: {steps:4d} | '
                    f'AvgReward: {avg_reward:+.4f} | '
                    f'Result: {result}\n')

        print(f'[Ep {self.episode_count:4d}] '
              f'Steps: {steps:4d} | '
              f'AvgReward: {avg_reward:+.4f} | '
              f'{result}')

        # Save numpy arrays every 10 episodes
        if self.episode_count % 10 == 0:
            np.save('episode_data/rewards.npy',
                    np.array(self.episode_rewards))
            np.save('episode_data/steps.npy',
                    np.array(self.episode_steps))
            print(f'>>> Episode data files saved '
                  f'(episode {self.episode_count})')

    # ──────────────────────────────────────────────────────────
    def _draw_hud(self):
        """Draw HUD with mode, controls, and danger warning."""
        font      = pygame.font.SysFont('monospace', 22, bold=True)
        font_warn = pygame.font.SysFont('monospace', 30, bold=True)

        # Semi-transparent background for HUD
        hud_bg = pygame.Surface((420, 175), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 150))
        self.display.blit(hud_bg, (0, 0))

        # Mode
        if self.human_input.is_takeover:
            mode_str   = "HUMAN TAKEOVER"
            mode_color = (255, 80, 80)
        else:
            mode_str   = "AI DRIVING"
            mode_color = (80, 255, 80)

        self.display.blit(
            font.render(mode_str, True, mode_color), (12, 10))
        self.display.blit(
            font.render(
                f"Steer:    {self.human_input.steer:+.3f}",
                True, (255, 255, 255)), (12, 36))
        self.display.blit(
            font.render(
                f"Throttle: {self.human_input.throttle:.2f}",
                True, (255, 255, 255)), (12, 60))
        self.display.blit(
            font.render(
                f"Brake:    {self.human_input.brake:.2f}",
                True, (255, 255, 255)), (12, 84))
        self.display.blit(
            font.render(
                f"Ep: {self.episode_count + 1}  "
                f"Step: {self.count}",
                True, (200, 200, 200)), (12, 108))
        self.display.blit(
            font.render(
                "SPACE=takeover  WASD=drive",
                True, (180, 180, 180)), (12, 132))
        self.display.blit(
            font.render(
                "Screen RED = DANGER, take over!",
                True, (255, 200, 0)), (12, 156))

        # Danger warning - flashing red overlay
        if self.danger_warning:
            self.warning_counter += 1
            if (self.warning_counter // 5) % 2 == 0:
                # Flash full screen red
                warn_overlay = pygame.Surface(
                    (1280, 720), pygame.SRCALPHA)
                warn_overlay.fill((255, 0, 0, 70))
                self.display.blit(warn_overlay, (0, 0))

                # Large warning text in center of screen
                warn_text = font_warn.render(
                    'DANGER!  PRESS SPACEBAR TO TAKE OVER',
                    True, (255, 255, 0))
                text_rect = warn_text.get_rect(center=(640, 380))

                # Background for text visibility
                text_bg = pygame.Surface(
                    (warn_text.get_width() + 20,
                     warn_text.get_height() + 10),
                    pygame.SRCALPHA)
                text_bg.fill((0, 0, 0, 180))
                self.display.blit(
                    text_bg,
                    (text_rect.x - 10, text_rect.y - 5))
                self.display.blit(warn_text, text_rect)
        else:
            self.warning_counter = 0

    # ──────────────────────────────────────────────────────────
    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.post_process()
                pygame.quit()
                sys.exit(0)
            self.human_input.process_event(event)

    # ──────────────────────────────────────────────────────────
    def destroy(self):
        """Safely destroy all CARLA actors."""
        for sensor in [self.seman_camera,
                       self.viz_camera,
                       self.collision_sensor]:
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
            except Exception:
                pass

        time.sleep(0.3)

        actors = [
            self.seman_camera,
            self.viz_camera,
            self.collision_sensor,
            self.ego_vehicle,
            self.obs1,
            self.obs2,
            self.obs3
        ]
        for actor in actors:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass

        self.seman_camera     = None
        self.viz_camera       = None
        self.collision_sensor = None
        self.ego_vehicle      = None
        self.obs1             = None
        self.obs2             = None
        self.obs3             = None

        # Reset camera for next episode
        self.camera_output = np.zeros(
            [720, 1280, 3], dtype=np.uint8)
        self.surface = None

        print('Actors destroyed - CARLA still running')

    # ──────────────────────────────────────────────────────────
    def post_process(self):
        """End of episode cleanup - keeps CARLA running."""
        try:
            self.destroy()
        except Exception as e:
            print(f'Cleanup warning: {e}')

    # ──────────────────────────────────────────────────────────
    def close(self):
        """Full shutdown - called only when training is done."""
        try:
            self.destroy()
        except Exception:
            pass
        try:
            if (hasattr(self, 'original_settings')
                    and self.original_settings):
                self.world.apply_settings(self.original_settings)
            print('Training complete - CARLA settings restored')
        except Exception as e:
            print(f'Close warning: {e}')
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────
    def obtain_observation(self):
        """Get state observation from semantic camera."""

        # Protect against empty camera
        if self.camera_output is None or np.sum(self.camera_output) == 0:
            self.camera_output = np.zeros(
                [720, 1280, 3], dtype=np.uint8)

        state_space = self.camera_output[:, :, 0].copy()
        if state_space.dtype != np.uint8:
            state_space = state_space.astype(np.uint8)

        state_space = cv2.resize(state_space, (WIDTH, HEIGHT))
        state_space = np.resize(
            state_space, (self.observation_size, 1))
        state_space = np.squeeze(state_space) / 255.0

        position_self = self.ego_vehicle.get_location()
        yaw_self      = self.ego_vehicle.get_transform().rotation.yaw
        position_obs1 = self.obs1.get_location()
        position_obs2 = self.obs2.get_location()
        position_obs3 = self.obs3.get_location()

        xa, ya, xb, yb, xc, yc, xd, yd = \
            self._to_corner_coordinate(
                position_self.x, position_self.y, yaw_self)

        xa1, ya1, xb1, yb1 = 337.4, 202.4, 339.6, 202.4
        xc1, yc1, xd1, yd1 = 339.6, 197.6, 337.4, 197.6
        xa2, ya2, xb2, yb2 = 333.9, 227.4, 336.1, 227.4
        xc2, yc2, xd2, yd2 = 336.1, 222.6, 333.9, 222.6
        xa3, ya3, xb3, yb3 = 337.4, 252.4, 339.6, 252.4
        xc3, yc3, xd3, yd3 = 339.6, 247.6, 337.4, 247.6

        if position_obs1.y - 4 < position_self.y < position_obs1.y + 4:
            state_corner_11 = self._sigmoid(
                np.clip(abs(xa1 - xa), 0, 10), 2.5)
            state_corner_12 = self._sigmoid(
                np.clip(abs(xa1 - xb), 0, 10), 2.5)
        else:
            state_corner_11 = state_corner_12 = 1

        if position_obs2.y - 4 < position_self.y < position_obs2.y + 4:
            state_corner_21 = self._sigmoid(
                np.clip(abs(xb2 - xa), 0, 10), 2.5)
            state_corner_22 = self._sigmoid(
                np.clip(abs(xb2 - xb), 0, 10), 2.5)
        else:
            state_corner_21 = state_corner_22 = 1

        if position_obs3.y - 4 < position_self.y < position_obs3.y + 4:
            state_corner_31 = self._sigmoid(
                np.clip(abs(xa3 - xa), 0, 10), 2.5)
            state_corner_32 = self._sigmoid(
                np.clip(abs(xa3 - xb), 0, 10), 2.5)
        else:
            state_corner_31 = state_corner_32 = 1

        state_left  = self._sigmoid(np.clip(340 - xb, 0, 10), 2)
        state_right = self._sigmoid(np.clip(xb - 332, 0, 10), 2)

        RIGHT = 1 if position_self.x < x_bench + 1.8 else 0
        if RIGHT:
            if position_self.y < y_bench + 25.0:
                state_front = self._sigmoid(
                    np.clip(yc2 - position_self.y - 2.6, 0, 25), 1)
            else:
                state_front = 1
        else:
            state_front = self._sigmoid(
                np.clip(yc3 - position_self.y - 2.4, 0, 25), 1)

        other_indicators = {
            'state_front':     state_front,
            'state_left':      state_left,
            'state_right':     state_right,
            'state_corner_11': state_corner_11,
            'state_corner_12': state_corner_12,
            'state_corner_21': state_corner_21,
            'state_corner_22': state_corner_22,
            'state_corner_31': state_corner_31,
            'state_corner_32': state_corner_32,
        }
        return state_space, other_indicators

    def obtain_real_observation(self):
        return self.camera_output[:, :, 0]

    # ──────────────────────────────────────────────────────────
    def _produce_vehicle_blueprint(self, color, x, y,
                                   vehicle='bmw'):
        if vehicle == 'bmw':
            bp = self.world.get_blueprint_library().filter(
                'vehicle.bmw.*')[0]
        elif vehicle == 'moto':
            bp = self.world.get_blueprint_library().filter(
                'vehicle.harley-davidson.*')[0]
        elif vehicle == 'bike':
            bp = self.world.get_blueprint_library().filter(
                'vehicle.diamondback.century.*')[0]
        elif vehicle == 'bus':
            bp = self.world.get_blueprint_library().filter(
                'vehicle.volkswagen.*')[0]
        else:
            bp = self.world.get_blueprint_library().filter(
                'vehicle.lincoln.*')[0]

        bp.set_attribute(
            'color',
            bp.get_attribute('color').recommended_values[color])

        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x  = x
        spawn_point.location.y  = y
        spawn_point.location.z += 0.3
        return bp, spawn_point

    def _produce_walker_blueprint(self, x, y):
        bp = self.world.get_blueprint_library().filter(
            'walker.*')[np.random.randint(2)]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x   = x
        spawn_point.location.y   = y
        spawn_point.location.z  += 0.1
        spawn_point.rotation.yaw = 0
        return bp, spawn_point

    def _toggle_camera(self):
        self.camera_transform_index = (
            (self.camera_transform_index + 1)
            % len(self.camera_transforms))

    def _dis_p_to_l(self, k, b, x, y):
        dis = abs((k * x - y + b) / math.sqrt(k * k + 1))
        return self._sigmoid(dis, 2)

    def _calculate_k_b(self, x1, y1, x2, y2):
        k = (y1 - y2) / (x1 - x2)
        b = (x1 * y2 - x2 * y1) / (x1 - x2)
        return k, b

    def _dis_p_to_p(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _to_corner_coordinate(self, x, y, yaw):
        xa = x + 2.64 * math.cos(yaw * math.pi / 180 - 0.43)
        ya = y + 2.64 * math.sin(yaw * math.pi / 180 - 0.43)
        xb = x + 2.64 * math.cos(yaw * math.pi / 180 + 0.43)
        yb = y + 2.64 * math.cos(yaw * math.pi / 180 + 0.43)
        xc = x + 2.64 * math.cos(
            yaw * math.pi / 180 - 0.43 + math.pi)
        yc = y + 2.64 * math.cos(
            yaw * math.pi / 180 - 0.43 + math.pi)
        xd = x + 2.64 * math.cos(
            yaw * math.pi / 180 + 0.43 + math.pi)
        yd = y + 2.64 * math.cos(
            yaw * math.pi / 180 + 0.43 + math.pi)
        return xa, ya, xb, yb, xc, yc, xd, yd

    def _sigmoid(self, x, theta):
        return 2.0 / (1 + math.exp(-theta * x)) - 1


# ──────────────────────────────────────────────────────────────
def find_weather_presets():
    rgx  = re.compile(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x))
            for x in presets]
