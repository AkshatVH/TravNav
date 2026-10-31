import argparse
import json
import os

import numpy as np
import mujoco
from PIL import Image

MAX_V = 0.2
MAX_W = 0.4
RATE = 4
WHEEL_RADIUS = 0.05
TRACK = 0.24
EPS = 1e-8
HERE = os.path.dirname(os.path.abspath(__file__))


def pd_controller(waypoint):
    dx, dy = waypoint[0], waypoint[1]
    dt = 1.0 / RATE
    if abs(dx) < EPS and abs(dy) < EPS:
        v, w = 0.0, 0.0
    elif abs(dx) < EPS:
        v, w = 0.0, np.sign(dy) * np.pi / (2 * dt)
    else:
        v, w = dx / dt, np.arctan(dy / dx) / dt
    return float(np.clip(v, 0.0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))


def wheel_speeds(v, w):
    left = (v - w * TRACK / 2.0) / WHEEL_RADIUS
    right = (v + w * TRACK / 2.0) / WHEEL_RADIUS
    return left, right


class StubPolicy:
    def push(self, frame):
        pass

    def set_goal(self, frame):
        pass

    def infer(self):
        steps = np.arange(1, 9, dtype=np.float32)
        return np.stack([steps, np.zeros_like(steps)], axis=-1)[None]


class OraclePolicy:
    def __init__(self, path, lookahead=0.5):
        self.path = np.array(path, dtype=np.float64)
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.lookahead = lookahead
        self.model = None
        self.data = None
        self.base_id = None

    def bind(self, model, data, base_id):
        self.model, self.data, self.base_id = model, data, base_id

    def push(self, frame):
        pass

    def set_goal(self, frame):
        pass

    def _project(self, p):
        best, best_s = 1e9, 0.0
        for i in range(len(self.seg_len)):
            a, b = self.path[i], self.path[i + 1]
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            d = np.linalg.norm(p - (a + t * ab))
            if d < best:
                best, best_s = d, self.cum[i] + t * self.seg_len[i]
        return best_s

    def _point_at(self, s):
        s = min(s, self.cum[-1])
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1))
        t = (s - self.cum[i]) / (self.seg_len[i] + 1e-9)
        return self.path[i] + t * (self.path[i + 1] - self.path[i])

    def infer(self):
        pos = self.data.xpos[self.base_id][:2]
        rot = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        target = self._point_at(self._project(pos) + self.lookahead)
        d = target - pos
        dx = np.cos(yaw) * d[0] + np.sin(yaw) * d[1]
        dy = -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]
        if dx < 0.05:
            dx = 0.0
        scale = MAX_V / RATE
        actions = np.zeros((1, 8, 2), dtype=np.float32)
        actions[0, :, 0] = dx / scale
        actions[0, :, 1] = dy / scale
        return actions


def contacts_with(model, data, robot_set, obstacle_set):
    found = set()
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = c.geom1, c.geom2
        if a in robot_set and b in obstacle_set:
            found.add(b)
        elif b in robot_set and a in obstacle_set:
            found.add(a)
    return found


def min_clearance(model, data, probe_geoms, obstacle_geoms, pos, distmax=1.0):
    best = distmax
    fromto = np.zeros(6)
    for g2 in obstacle_geoms:
        if np.linalg.norm(data.geom_xpos[g2][:2] - pos) > 1.5:
            continue
        for g1 in probe_geoms:
            d = mujoco.mj_geomDistance(model, data, g1, g2, distmax, fromto)
            if d < best:
                best = d
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=os.path.join(HERE, "world.xml"))
    parser.add_argument("--path-json", default=os.path.join(HERE, "path.json"))
    parser.add_argument("--ckpt", default="model_weights/nomad.pth")
    parser.add_argument("--goal", default=None)
    parser.add_argument("--render-goal", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"), default=None)
    parser.add_argument("--goal-out", default="goal.png")
    parser.add_argument("--waypoint", type=int, default=2)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--save-frames", default=None)
    parser.add_argument("--save-map", default=None)
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--stop-on-collision", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()

    with open(args.path_json) as f:
        meta = json.load(f)
    goal_x = args.goal_x if args.goal_x is not None else meta["goal_x"]

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    if args.render_goal is not None:
        gx, gy, gyaw = args.render_goal
        yaw = np.radians(gyaw)
        qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
        data.qpos[qadr:qadr + 3] = [gx, gy, 0.05]
        data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
        goal_renderer = mujoco.Renderer(model, height=480, width=640)
        goal_renderer.update_scene(data, camera="front")
        Image.fromarray(goal_renderer.render()).save(args.goal_out)
        goal_renderer.close()
        print(f"saved goal image from pose x={gx} y={gy} yaw={gyaw} deg to {args.goal_out}")
        return

    if args.oracle:
        policy = OraclePolicy(meta["path"])
        policy.bind(model, data, base_id)
    elif args.stub:
        policy = StubPolicy()
    else:
        from nomad_policy import NoMaDPolicy

        policy = NoMaDPolicy(args.ckpt)
        if args.goal:
            policy.set_goal(np.array(Image.open(args.goal).convert("RGB")))

    renderer = mujoco.Renderer(model, height=480, width=640)

    robot_geoms = {
        g for g in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[g]] == base_id
    }
    obstacle_geoms = {
        g for g in range(model.ngeom)
        if (model.geom(g).name or "").startswith(("obs_", "clutter_", "wall_"))
    }
    probe_geoms = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("chassis", "left_wheel_geom", "right_wheel_geom")
    ]

    mujoco.mj_forward(model, data)
    movable_geoms = [
        g for g in obstacle_geoms
        if not (model.geom(g).name or "").startswith("wall_")
    ]
    initial_pos = {g: data.geom_xpos[g][:2].copy() for g in movable_geoms}

    if args.save_map:
        renderer.update_scene(data, camera="topdown")
        Image.fromarray(renderer.render()).save(args.save_map)

    physics_steps = int(round((1.0 / RATE) / model.opt.timestep))
    scale = MAX_V / RATE
    frames = []
    last_contact = {}
    collision_events = 0
    hit_names = set()
    clearance = 1.0
    travelled = 0.0
    last_pos = data.xpos[base_id][:2].copy()
    reached = False
    stopped_on_collision = False
    steps_done = 0

    for step in range(args.steps):
        renderer.update_scene(data, camera="front")
        obs_frame = renderer.render().copy()
        policy.push(obs_frame)

        actions = policy.infer()
        waypoint = actions[0][args.waypoint].copy() * scale
        v, w = pd_controller(waypoint)
        left, right = wheel_speeds(v, w)
        data.ctrl[0] = left
        data.ctrl[1] = right

        hit_this_step = False
        touch_this_step = False
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)
            hits = contacts_with(model, data, robot_geoms, obstacle_geoms)
            for g in hits:
                touch_this_step = True
                if data.time - last_contact.get(g, -1e9) > 0.5:
                    collision_events += 1
                    hit_names.add(model.geom(g).name)
                    hit_this_step = True
                last_contact[g] = data.time

        pos = data.xpos[base_id][:2].copy()
        travelled += float(np.linalg.norm(pos - last_pos))
        last_pos = pos
        clearance = min(clearance, min_clearance(model, data, probe_geoms, obstacle_geoms, pos))
        steps_done = step + 1

        if args.save_frames:
            frames.append(obs_frame)

        print(f"step={step} pos=({pos[0]:.2f},{pos[1]:.2f}) v={v:.3f} w={w:.3f} hit={int(hit_this_step)} touch={int(touch_this_step)}")

        if pos[0] >= goal_x - 0.1:
            reached = True
            break
        if args.stop_on_collision and hit_this_step:
            stopped_on_collision = True
            break

    if args.save_frames and frames:
        import imageio

        imageio.mimsave(args.save_frames, frames, fps=RATE)

    renderer.close()

    moved = {
        model.geom(g).name: float(np.linalg.norm(data.geom_xpos[g][:2] - initial_pos[g]))
        for g in movable_geoms
    }
    moved = {k: v for k, v in moved.items() if v > 0.02}

    print("---- summary ----")
    print(f"reached_goal      : {reached}")
    print(f"steps             : {steps_done} ({steps_done / RATE:.1f} s)")
    print(f"distance_travelled: {travelled:.2f} m")
    print(f"collision_episodes : {collision_events}")
    print(f"objects_hit       : {sorted(hit_names)}")
    print(f"objects_moved     : {len(moved)} (max {max(moved.values(), default=0.0) * 100:.0f} cm)")
    print(f"min_clearance     : {clearance * 100:.1f} cm")
    print(f"gap_width         : {meta['gap'] * 100:.0f} cm")
    if stopped_on_collision:
        print("stopped early on collision")


if __name__ == "__main__":
    main()