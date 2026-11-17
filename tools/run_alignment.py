from vint_train.data.vint_dataset import ViNT_Dataset
from tools.check_dataset_maps import run, available_indices


dataset = ViNT_Dataset(
    data_folder="./train/vint_train/data/go_stanford",
    data_split_folder="train/vint_train/data/data_splits/go_stanford/train",
    dataset_name="go_stanford",
    image_size=(96, 96),
    waypoint_spacing=1,
    min_dist_cat=0,
    max_dist_cat=20,
    min_action_distance=0,
    max_action_distance=20,
    negative_mining=True,
    len_traj_pred=8,
    learn_angle=True,
    context_size=5,
    context_type="temporal",
    end_slack=0,
    goals_per_obs=1,
    normalize=True,
    goal_type="image",
    teacher_dir="./tools/out/data",
    teacher_mode="dino",
)

pool = available_indices(dataset, "./tools/out/data", "go_stanford")
run(dataset, n=20, out_prefix="alignment_check", indices=pool)

""" item = dataset[pool[0]]
print(dataset.index_to_data[pool[0]])
print([a for a in dir(dataset) if "path" in a or "index" in a])

print(len(dataset), len(item))
for k, x in enumerate(item):
    print(k, getattr(x, "shape", x)) """