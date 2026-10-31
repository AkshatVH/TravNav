import argparse
import json
import os

import numpy as np
from PIL import Image

TEX = 512


def fbm(rng, octaves=((4, 0.5), (16, 0.3), (64, 0.2)), aspect=(1, 1)):
    out = np.zeros((TEX, TEX))
    for cells, amp in octaves:
        small = rng.random((max(2, cells // aspect[0]), max(2, cells // aspect[1])))
        img = Image.fromarray((small * 255).astype(np.uint8)).resize(
            (TEX, TEX), Image.BICUBIC
        )
        out += amp * np.asarray(img) / 255.0
    return out


def shade(base, variation, rng, grain=0.02, noise=None):
    n = fbm(rng) if noise is None else noise
    n = (n - n.mean()) / (n.std() + 1e-6)
    img = np.ones((TEX, TEX, 3)) * np.array(base)
    img = img * (1.0 + variation * n[..., None])
    img += rng.normal(0, grain, img.shape)
    return np.clip(img, 0, 1)


def save(img, path):
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def make_textures(outdir, seed):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)

    floor = shade((0.52, 0.50, 0.47), 0.10, rng, grain=0.015)
    for line in (0, TEX // 2):
        floor[max(0, line - 2): line + 2, :, :] *= 0.62
        floor[:, max(0, line - 2): line + 2, :] *= 0.62
    save(floor, os.path.join(outdir, "floor.png"))

    save(shade((0.55, 0.32, 0.29), 0.04, rng, grain=0.012), os.path.join(outdir, "wall.png"))

    streak = fbm(rng, octaves=((2, 0.4), (8, 0.3), (32, 0.3)), aspect=(1, 8))
    save(shade((0.60, 0.46, 0.31), 0.10, rng, grain=0.02, noise=streak),
         os.path.join(outdir, "cardboard.png"))

    x = np.linspace(0, 1, TEX)[None, :].repeat(TEX, axis=0)
    warp = fbm(rng, octaves=((2, 0.6), (6, 0.4)))
    grain = np.sin((x * 38 + warp * 9) * np.pi) * 0.5 + 0.5
    wood = np.ones((TEX, TEX, 3)) * np.array((0.42, 0.30, 0.19))
    wood = wood * (0.78 + 0.34 * grain[..., None]) + rng.normal(0, 0.012, (TEX, TEX, 3))
    save(np.clip(wood, 0, 1), os.path.join(outdir, "wood.png"))

    save(shade((0.52, 0.30, 0.19), 0.07, rng, grain=0.01), os.path.join(outdir, "plastic.png"))

    weave = (np.indices((TEX, TEX)).sum(axis=0) // 3 % 2) * 0.04
    fabric = shade((0.52, 0.30, 0.29), 0.08, rng, grain=0.02)
    save(np.clip(fabric + weave[..., None], 0, 1), os.path.join(outdir, "fabric.png"))


def dist_to_polyline(p, poly):
    best = 1e9
    for a, b in zip(poly[:-1], poly[1:]):
        a = np.array(a)
        b = np.array(b)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        best = min(best, np.linalg.norm(p - (a + t * ab)))
    return best


def build_course(rng, args):
    half = args.width / 2.0
    gap = args.gap
    geoms = []
    path = [(args.start_x, 0.0)]
    gc = 0.0
    xs = np.arange(args.first_row, args.goal_x - 0.9, args.row_spacing)
    counter = [0]

    def pick_height(base_width):
        top = max(args.min_height, min(args.max_height, 3.0 * base_width))
        return float(rng.uniform(args.min_height, top))

    def add(kind, mat, x, y, sx, sy, height):
        counter[0] += 1
        geoms.append(
            dict(name=f"obs_{counter[0]}", kind=kind, mat=mat, x=x, y=y,
                 sx=sx, sy=sy, h=height, yaw=0.0)
        )

    def fill(a, b, x_row):
        cursor = a
        spacing = 0.03
        while b - cursor > 0.06:
            w = rng.uniform(0.18, 0.42)
            last = b - (cursor + w) < 0.14
            if last:
                y0, y1 = cursor, b
            else:
                y0, y1 = cursor, cursor + w - spacing
            width = y1 - y0
            depth = rng.uniform(0.16, 0.42)
            xc = x_row + rng.uniform(-0.12, 0.12)
            yc = (y0 + y1) / 2.0
            kind = rng.choice(["box", "box", "cyl", "bag"])
            if kind == "box":
                mat = rng.choice(["cardboard", "wood"])
                add("box", mat, xc, yc, depth / 2.0, width / 2.0, pick_height(min(width, depth)))
            elif kind == "cyl":
                r = width / 2.0
                add("cyl", rng.choice(["plastic", "plastic", "wood"]), xc, yc, r, r, pick_height(width))
            else:
                add("box", "fabric", xc, yc, depth / 2.0, width / 2.0, pick_height(min(width, depth)))
            cursor = y1 + spacing if not last else b
            if last:
                break

    for x in xs:
        lim = half - gap / 2.0 - 0.10
        gc = float(np.clip(gc + rng.uniform(-args.shift, args.shift), -lim, lim))
        path.append((float(x) - 0.5, gc))
        path.append((float(x) + 0.5, gc))
        fill(-half, gc - gap / 2.0, x)
        fill(gc + gap / 2.0, half, x)
    path.append((args.goal_x, 0.0))

    placed = 0
    tries = 0
    while placed < args.clutter and tries < 2000:
        tries += 1
        x = rng.uniform(1.0, args.goal_x - 0.6)
        y = rng.uniform(-half + 0.12, half - 0.12)
        if dist_to_polyline(np.array([x, y]), path) < 0.45:
            continue
        if np.min(np.abs(xs - x)) < 0.6:
            continue
        if any(np.hypot(x - c["x"], y - c["y"]) < 0.3 for c in geoms if c["name"].startswith("clutter_")):
            continue
        s = rng.uniform(0.07, 0.14)
        counter[0] += 1
        geoms.append(
            dict(name=f"clutter_{counter[0]}", kind="box",
                 mat=rng.choice(["cardboard", "plastic", "fabric"]),
                 x=float(x), y=float(y), sx=s, sy=s * rng.uniform(0.7, 1.4),
                 h=float(rng.uniform(args.min_height, args.min_height + 0.15)), yaw=float(rng.uniform(0, 90)))
        )
        placed += 1
    return geoms, path


DENSITY = {"cardboard": 80.0, "wood": 220.0, "plastic": 120.0, "fabric": 90.0}


def mass_of(g, max_mass):
    h = g["h"]
    if g["kind"] == "box":
        vol = (2 * g["sx"]) * (2 * g["sy"]) * h
    else:
        vol = np.pi * g["sx"] ** 2 * h
    return float(np.clip(vol * DENSITY[g["mat"]], 0.15, max_mass))


def geom_xml(g, movable, max_mass):
    h = g["h"]
    if g["kind"] == "box":
        shape = f'type="box" size="{g["sx"]:.3f} {g["sy"]:.3f} {h / 2:.3f}" euler="0 0 {g["yaw"]:.1f}"'
    else:
        shape = f'type="cylinder" size="{g["sx"]:.3f} {h / 2:.3f}"'
    pos = f'{g["x"]:.3f} {g["y"]:.3f} {h / 2:.3f}'
    if not movable:
        return f'    <geom name="{g["name"]}" {shape} material="m_{g["mat"]}" pos="{pos}"/>'
    mass = mass_of(g, max_mass)
    return (
        f'    <body name="b_{g["name"]}" pos="{pos}">\n'
        f'      <freejoint/>\n'
        f'      <geom name="{g["name"]}" {shape} material="m_{g["mat"]}" mass="{mass:.3f}" friction="0.5 0.005 0.0001"/>\n'
        f'    </body>'
    )


def write_world(path_xml, geoms, args):
    half = args.width / 2.0
    length = args.goal_x + 1.2
    cx = (length - 0.8) / 2.0 - 0.0
    lines = []
    lines.append(f"""<mujoco model="obstacle_course">
  <compiler angle="degree" texturedir="textures"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.38 0.38 0.38" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.72 0.72 0.70" rgb2="0.86 0.86 0.84" width="256" height="256"/>
    <texture name="t_floor" type="2d" file="floor.png"/>
    <texture name="t_wall" type="2d" file="wall.png"/>
    <texture name="t_cardboard" type="2d" file="cardboard.png"/>
    <texture name="t_wood" type="2d" file="wood.png"/>
    <texture name="t_plastic" type="2d" file="plastic.png"/>
    <texture name="t_fabric" type="2d" file="fabric.png"/>
    <material name="m_floor" texture="t_floor" texuniform="true" texrepeat="1 1" specular="0.05" shininess="0.1" reflectance="0.01"/>
    <material name="m_wall" texture="t_wall" texuniform="true" texrepeat="0.5 0.5" specular="0.02" shininess="0.05"/>
    <material name="m_cardboard" texture="t_cardboard" texuniform="true" texrepeat="1 1" specular="0.02" shininess="0.05"/>
    <material name="m_wood" texture="t_wood" texuniform="true" texrepeat="1 1" specular="0.1" shininess="0.2"/>
    <material name="m_plastic" texture="t_plastic" texuniform="true" texrepeat="1 1" specular="0.2" shininess="0.3"/>
    <material name="m_fabric" texture="t_fabric" texuniform="true" texrepeat="2 2" specular="0.01" shininess="0.02"/>
  </asset>
  <worldbody>
    <light pos="2 0 2.4" dir="0 0 -1" diffuse="0.45 0.45 0.43" castshadow="true"/>
    <light pos="6 0 2.4" dir="0 0 -1" diffuse="0.45 0.45 0.43" castshadow="true"/>
    <light pos="10 0 2.4" dir="0 0 -1" diffuse="0.45 0.45 0.43" castshadow="true"/>
    <light pos="14 0 2.4" dir="0 0 -1" diffuse="0.45 0.45 0.43" castshadow="true"/>
    <geom name="floor" type="plane" size="{length / 2 + 1:.2f} {half + 1:.2f} 0.1" pos="{cx:.2f} 0 0" material="m_floor" friction="0.5 0.005 0.0001"/>
    <geom name="wall_left" type="box" size="{length / 2:.2f} 0.05 0.6" pos="{cx:.2f} {half + 0.05:.2f} 0.6" material="m_wall"/>
    <geom name="wall_right" type="box" size="{length / 2:.2f} 0.05 0.6" pos="{cx:.2f} {-half - 0.05:.2f} 0.6" material="m_wall"/>
    <geom name="wall_back" type="box" size="0.05 {half + 0.1:.2f} 0.6" pos="-0.8 0 0.6" material="m_wall"/>
    <geom name="wall_end" type="box" size="0.05 {half + 0.1:.2f} 0.6" pos="{args.goal_x + 1.2:.2f} 0 0.6" material="m_wall"/>
    <geom name="goal_tape" type="box" size="0.03 {half:.2f} 0.001" pos="{args.goal_x:.2f} 0 0.002" rgba="0.72 0.66 0.30 1" contype="0" conaffinity="0"/>
""")
    for g in geoms:
        lines.append(geom_xml(g, not args.static, args.max_mass))
    lines.append(f"""
    <body name="base" pos="{args.start_x} 0 0.05">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="0.15 0.1 0.03" pos="0 0 0.01" mass="3" rgba="0.18 0.2 0.22 1"/>
      <geom name="caster_front" type="sphere" size="0.02" pos="0.12 0 -0.03" mass="0.05" friction="0.001 0.001 0.001" priority="1" condim="1"/>
      <geom name="caster_rear" type="sphere" size="0.02" pos="-0.12 0 -0.03" mass="0.05" friction="0.001 0.001 0.001" priority="1" condim="1"/>
      <camera name="front" pos="0.15 0 0.2" xyaxes="0 -1 0 0 0 1" fovy="75"/>
      <body name="left_wheel" pos="0 0.12 0">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom name="left_wheel_geom" type="cylinder" size="0.05 0.02" euler="90 0 0" mass="0.3" friction="1.5 0.005 0.0001" rgba="0.08 0.08 0.08 1"/>
      </body>
      <body name="right_wheel" pos="0 -0.12 0">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom name="right_wheel_geom" type="cylinder" size="0.05 0.02" euler="90 0 0" mass="0.3" friction="1.5 0.005 0.0001" rgba="0.08 0.08 0.08 1"/>
      </body>
    </body>

    <camera name="topdown" pos="{cx:.2f} 0 12" xyaxes="1 0 0 0 1 0" fovy="55"/>
  </worldbody>
  <actuator>
    <velocity name="left_motor" joint="left_wheel_joint" kv="0.5" ctrlrange="-30 30" forcerange="-2 2"/>
    <velocity name="right_motor" joint="right_wheel_joint" kv="0.5" ctrlrange="-30 30" forcerange="-2 2"/>
  </actuator>
</mujoco>
""")
    with open(path_xml, "w") as f:
        f.write("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="world.xml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap", type=float, default=0.6)
    parser.add_argument("--width", type=float, default=3.2)
    parser.add_argument("--goal-x", type=float, default=14.0)
    parser.add_argument("--start-x", type=float, default=0.3)
    parser.add_argument("--first-row", type=float, default=2.0)
    parser.add_argument("--row-spacing", type=float, default=1.5)
    parser.add_argument("--shift", type=float, default=0.7)
    parser.add_argument("--clutter", type=int, default=18)
    parser.add_argument("--min-height", type=float, default=0.4)
    parser.add_argument("--max-height", type=float, default=0.9)
    parser.add_argument("--max-mass", type=float, default=2.0)
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    make_textures(os.path.join(out_dir, "textures"), args.seed)
    rng = np.random.default_rng(args.seed)
    geoms, path = build_course(rng, args)
    write_world(args.out, geoms, args)
    with open(os.path.join(out_dir, "path.json"), "w") as f:
        json.dump({"path": path, "gap": args.gap, "goal_x": args.goal_x}, f)
    print(f"wrote {args.out} with {len(geoms)} obstacles, gap={args.gap} m, seed={args.seed}, shift = {args.shift}")


if __name__ == "__main__":
    main()