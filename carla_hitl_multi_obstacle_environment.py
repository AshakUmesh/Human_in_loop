import random
import time
import math
import numpy as np
import carla


class MultiObstacleScenario:
    """
    CARLA multi-obstacle HITL environment.

    New obstacle types:
    1) parked / wrong-way car
    2) oncoming vehicle in ego lane
    3) pedestrian jaywalking across road
    4) pedestrian on zebra crossing
    5) slow walker along crosswalk edge
    """

    def __init__(self, client, frame=25):
        self.client = client
        self.world = client.get_world()
        self.frame = frame
        self.actors = []
        self.walker_controllers = []
        self.ego_vehicle = None
        self.original_settings = None

    # --------------------------------------------------
    # episode reset
    # --------------------------------------------------
    def reset(self):
        self.destroy()
        self._apply_sync_once()

        self._spawn_ego()
        self._spawn_static_wrong_way_car()
        self._spawn_oncoming_vehicle()
        self._spawn_crossing_pedestrian()
        self._spawn_crosswalk_pedestrian()
        self._spawn_sidewalk_walker()

        for _ in range(5):
            self.world.tick()
            time.sleep(0.03)

        return self.get_state()

    # --------------------------------------------------
    # world settings
    # --------------------------------------------------
    def _apply_sync_once(self):
        if self.original_settings is None:
            self.original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame
            self.world.apply_settings(settings)

    # --------------------------------------------------
    # ego
    # --------------------------------------------------
    def _spawn_ego(self):
        bp = self.world.get_blueprint_library().filter('vehicle.tesla.*')[0]
        spawn = carla.Transform(
            carla.Location(x=335.0, y=200.0, z=0.3),
            carla.Rotation(yaw=90)
        )
        self.ego_vehicle = self.world.spawn_actor(bp, spawn)
        self.actors.append(self.ego_vehicle)
        self.ego_vehicle.set_target_velocity(carla.Vector3D(0, 4.0, 0))

    # --------------------------------------------------
    # obstacle 1: wrong-way parked car
    # --------------------------------------------------
    def _spawn_static_wrong_way_car(self):
        bp = self.world.get_blueprint_library().filter('vehicle.audi.*')[0]
        spawn = carla.Transform(
            carla.Location(x=336.8, y=228.0, z=0.3),
            carla.Rotation(yaw=-90)
        )
        actor = self.world.spawn_actor(bp, spawn)
        self.actors.append(actor)

    # --------------------------------------------------
    # obstacle 2: oncoming car
    # --------------------------------------------------
    def _spawn_oncoming_vehicle(self):
        bp = self.world.get_blueprint_library().filter('vehicle.bmw.*')[0]
        spawn = carla.Transform(
            carla.Location(x=335.0, y=255.0, z=0.3),
            carla.Rotation(yaw=-90)
        )
        actor = self.world.spawn_actor(bp, spawn)
        actor.set_target_velocity(carla.Vector3D(0, -3.0, 0))
        self.actors.append(actor)

    # --------------------------------------------------
    # obstacle 3: jaywalking pedestrian
    # --------------------------------------------------
    def _spawn_crossing_pedestrian(self):
        bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
        spawn = carla.Transform(
            carla.Location(x=332.5, y=218.0, z=0.1)
        )
        walker = self.world.spawn_actor(bp, spawn)
        self.actors.append(walker)

        control = carla.WalkerControl(
            direction=carla.Vector3D(x=1.0, y=0.0, z=0.0),
            speed=1.2
        )
        walker.apply_control(control)

    # --------------------------------------------------
    # obstacle 4: crosswalk pedestrian
    # --------------------------------------------------
    def _spawn_crosswalk_pedestrian(self):
        bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
        spawn = carla.Transform(
            carla.Location(x=333.0, y=240.0, z=0.1)
        )
        walker = self.world.spawn_actor(bp, spawn)
        self.actors.append(walker)

        control = carla.WalkerControl(
            direction=carla.Vector3D(x=1.0, y=0.0, z=0.0),
            speed=0.8
        )
        walker.apply_control(control)

    # --------------------------------------------------
    # obstacle 5: pedestrian walking along crosswalk
    # --------------------------------------------------
    def _spawn_sidewalk_walker(self):
        bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
        spawn = carla.Transform(
            carla.Location(x=340.5, y=235.0, z=0.1)
        )
        walker = self.world.spawn_actor(bp, spawn)
        self.actors.append(walker)

        control = carla.WalkerControl(
            direction=carla.Vector3D(x=0.0, y=1.0, z=0.0),
            speed=0.5
        )
        walker.apply_control(control)

    # --------------------------------------------------
    # step
    # --------------------------------------------------
    def step(self, steer):
        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1, 1))
        control.throttle = 0.35
        self.ego_vehicle.apply_control(control)

        self.world.tick()

        reward, done = self._compute_reward_done()
        state = self.get_state()
        return state, reward, done

    # --------------------------------------------------
    # reward shaping for diverse hazards
    # --------------------------------------------------
    def _compute_reward_done(self):
        ego_loc = self.ego_vehicle.get_location()
        reward = 0.2
        done = False

        # penalty for lane deviation
        lane_center_x = 335.0
        reward -= abs(ego_loc.x - lane_center_x) * 1.5

        # distance penalties for all dynamic actors
        for actor in self.actors[1:]:
            if not actor.is_alive:
                continue
            loc = actor.get_location()
            d = math.sqrt((ego_loc.x - loc.x) ** 2 + (ego_loc.y - loc.y) ** 2)
            if d < 3.0:
                reward -= (3.0 - d) * 3.0
            if d < 1.2:
                reward -= 10
                done = True

        # success zone
        if ego_loc.y > 270:
            reward += 10
            done = True

        return float(np.clip(reward, -10, 10)), done

    # --------------------------------------------------
    # simple state
    # --------------------------------------------------
    def get_state(self):
        ego = self.ego_vehicle.get_location()
        state = [ego.x, ego.y]

        for actor in self.actors[1:6]:
            loc = actor.get_location()
            state.extend([loc.x - ego.x, loc.y - ego.y])

        return np.array(state, dtype=np.float32)

    # --------------------------------------------------
    # cleanup
    # --------------------------------------------------
    def destroy(self):
        for actor in self.actors:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actors = []

    def close(self):
        self.destroy()
        if self.original_settings is not None:
            self.world.apply_settings(self.original_settings)
