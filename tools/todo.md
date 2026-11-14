# TravNav: What To Do, Where, In What Order

Current state: the seed-prototype teacher works on one image.
Goal of this guide: turn it into saved teacher maps for every frame, then build the student side (decoder, tokenizer, 11-token encoder, loss) so training can start.

This guide describes the changes, not the code. Each section says which file, what to change, and how to know it worked.

---

## Part A. Teacher pipeline

### A1. `tools/make_teacher_maps.py` (modify)

**Why this file:** it is the script that loops over trajectories, batches frames, calls the teacher and writes one `.npy` per frame. You keep its structure and only change what the teacher is.

#### A1.1 Resolution

- Change the `Resize` target in the transform from `(224, 224)` to `(448, 448)`.
- 448 / 14 = 32, so the patch grid becomes 32x32.
- `teacher_map` already computes `side` from `frames.shape[-1] // 14`, so no other edit is needed there.
- The final bilinear resize to `64x64` stays. It now upsamples 32 to 64 instead of 16 to 64, which is less blurry.

#### A1.2 Device

- Replace the hardcoded `"cuda"` with a check: use CUDA if available, otherwise CPU.
- Pass that device to `load_dino` and `load_head`, and use it when moving the batch.

#### A1.3 Arguments

- `--head`: make it optional or delete it. There are no weights to load.
- Add `--limit N`: stop after N frames in total (or N per trajectory). You will use it for timing and quality tests.
- Add `--seed_rows` (default 3) and `--sigma` (default 2.0) so you can adjust the seed region and threshold without editing code.
- `--out` must already include the dataset name, so the result is `<out>/<traj_name>/<frame_index>.npy`.

#### A1.4 `load_head` becomes your prototype method

Keep the contract the rest of the script expects:

| | Shape | Notes |
|---|---|---|
| Input | `(B, C, 32, 32)` | DINOv2 patch tokens reshaped into a grid |
| Output | `(B, 1, 32, 32)` | values in `[0, 1]`, on the same device |

`load_head` returns a callable (a plain function or a parameter-free `nn.Module`). For each image `i` in the batch, do these steps in order:

1. **Flatten and normalise.** Reshape the grid to `(N, C)` with `N = 32*32`, then L2-normalise along `C`. The plan's script does not normalise the backbone output, and your first script did, so do it here.
2. **Seed mask.** Build a boolean `(32, 32)` mask:
   - rows: the last `seed_rows` rows (3 of 32 is the equivalent of 4 of 40)
   - columns: `32 // 3` up to `2 * 32 // 3` (the middle third)
3. **Prototype.** Take the features inside the seed mask, average them, and L2-normalise the result.
4. **Similarity.** Dot product of every patch feature with the prototype, giving `N` values, reshaped to `(32, 32)`.
5. **Threshold.** Take the similarities inside the seed mask and compute `mean - sigma * std`.
6. **Connected component.** Binarise with the threshold, move to CPU/numpy, run `scipy.ndimage.label`, collect the labels that appear inside the seed mask (ignore label 0), and keep only pixels with those labels.
7. **Soft map.** Rescale the similarity so that:
   - `threshold` maps to 0
   - the seed mean similarity maps to 1
   - values outside are clipped to `[0, 1]`
8. **Combine.** Multiply the soft map by the connected mask. The result is high where the patch looks like floor and is connected to the seed, and zero elsewhere.
9. **Stack.** Collect the `B` maps into one `(B, 1, 32, 32)` float tensor on the right device.

Why the soft map times the connected mask: the soft part keeps "how floor-like", the mask part removes floor-coloured regions that are not connected to where the robot actually is (for example a table surface far away).

Also record, per image, the **seed spread** (std of similarities inside the seed mask). You will use it in A2 as a quality signal. Simplest way: keep a list on the head object that `make_teacher_maps.py` can read after each batch and write to a CSV (`traj, frame, seed_std, seed_mean`).

#### A1.5 Things to verify in this file

- The frame index in the output filename comes from `f.stem`. The dataset class uses the same integer when it loads the current observation, so these must match exactly (no zero padding differences).
- Frames in your datasets are probably not square. `Resize((448, 448))` squashes them. That is fine for the teacher, but remember the map is therefore in the squashed frame. Your student decoder sees the original image at its own resolution, so the map is compared as a normalised 64x64 grid, which is consistent as long as you always squash the same way.
- `torch.hub.load` downloads from GitHub. If your machine has no internet at run time, load the local `vit_small` as in your first script instead.

**Done when:** running with `--limit 16` writes 16 `.npy` files, each `(64, 64)`, float16, values in `[0, 1]`.

---

### A2. `tools/check_teacher_maps.py` (new)

**Why:** your method fails silently when the seed region is not floor. You need to see failures before training on them.

What the script does:

1. **Count check.** For every trajectory folder, compare the number of `.jpg` files with the number of `.npy` files. Print any trajectory where they differ.
2. **Value check.** For a random sample of maps, assert shape `(64, 64)`, dtype float16, min >= 0, max <= 1. Print the fraction of maps that are almost all zero (for example, mean below 0.02) and almost all one (mean above 0.98). Both extremes are suspicious.
3. **Visual grid.** Randomly pick about 20 frames spread across different trajectories (not 20 from one). For each, plot the frame, the map, and the frame with the map overlaid. Save one PNG (for example 4 rows by 5 columns).
4. **Seed spread ranking.** Read the CSV from A1.4, sort by `seed_std` descending, and show the 10 worst frames as a second PNG. These are where the seed is probably not pure floor.

How to judge:

- Good: the map covers floor, stops at walls and furniture, and the bottom-middle of the frame is bright.
- Bad: bright wall, bright person, or all-black map. If more than roughly one in ten sampled frames is bad, fix the teacher before generating everything.

Possible fixes, in order of effort:

1. Change `--seed_rows` or the column range.
2. Change `--sigma` (a smaller value gives a tighter mask).
3. Add a filter: if `seed_std` is above a chosen cutoff, flag the frame and write a `valid` flag file or a CSV entry. The dataset hook (Part C) can then skip loss on flagged frames or the frames can be dropped.

**Done when:** the grid looks right on frames from at least 5 different trajectories and the count check passes.

---

### A3. Timing and full run

1. Run `make_teacher_maps.py --limit 16` and note seconds per frame.
2. Multiply by the total number of frames in your indoor datasets to estimate the full time.
3. If the estimate is too long:
   - lower `--batch` only if you hit out-of-memory (it does not speed things up)
   - process every 2nd frame (add a `--stride 2` argument) and train only on those frames
4. Run the full job in the background (`nohup` or `tmux`), one dataset at a time, writing to `/data/teacher_maps/<dataset_name>/`.
5. Re-run `check_teacher_maps.py` on the finished output.

**Done when:** every trajectory in the train and test splits has a matching `.npy` for every frame the dataset can request.

---

## Part B. Student modules

### B1. `train/vint_train/models/travnav/trav_modules.py` (new)

Create the `travnav` folder with an empty `__init__.py` if the other model folders have one. The file contains five things, all defined in Day 4b of your plan:

| Item | Purpose | Input | Output |
|---|---|---|---|
| `TraversabilityDecoder` | Predicts the map from the encoder's multi-scale features | stride 8, 16, 32 feature maps of the current frame | `(B, 1, 64, 64)` in `[0, 1]` |
| `TraversabilityTokenizer` | Compresses the map into tokens | `(B, 1, 64, 64)` | `(B, 4, 256)` |
| `distil_loss` | `L_distil` | prediction, target, mode | scalar |
| `rasterize_path` | Stand-in teacher from future waypoints | `(B, T, 2)` | `(B, 1, 64, 64)` |
| `get_teacher_map` | Chooses real teacher or stand-in | mode, loaded teacher batch, actions | `(B, 1, 64, 64)` |

What to set yourself:

- `in_channels` for the decoder is a list of three channel counts, in order `[stride 8, stride 16, stride 32]`. Use the numbers you wrote down on Day 1 from `extract_endpoints`. Do not copy them from memory.
- `tokenizer` `grid=2` gives `2*2 = 4` tokens, which matches `num_trav_tokens: 4`. If you change one, change the other.
- `dim=256` must equal NoMaD's `obs_encoding_size`.

Because you now have a real teacher, `rasterize_path` is only a fallback. You may still keep it (it is short) for debugging before the maps finish.

### B2. `tests/test_trav_modules.py` (new)

Tests to write:

- Decoder: random `s8, s16, s32` with the real shapes for a 96x96 input gives `(B, 1, 64, 64)`, min >= 0, max <= 1.
- Tokenizer: `(B, 1, 64, 64)` in, `(B, 4, 256)` out.
- Rasteriser: random waypoints in, `(B, 1, 64, 64)` out.
- `get_teacher_map`: both modes return the same shape and dtype.
- `distil_loss`: returns a scalar and is zero when prediction equals target.

**Done when:** all tests pass with `pytest`.

---

## Part C. Encoder and dataset

### C1. `train/vint_train/models/travnav/travnav_vint.py` (new)

Subclass `NoMaD_ViNT`. Open the parent file first and copy its `forward`. Change only these points:

1. **Constructor arguments.** Add `num_trav_tokens` (4), `trav_dropout` (0.1), `use_trav` (True) and the decoder `in_channels`.
2. **New modules.** Create the decoder, the tokenizer and a type embedding `nn.Embedding(3, obs_encoding_size)`.
3. **Sequence length.** Define `self.num_tokens = (context_size + 1) + num_trav_tokens + 1`. With `context_size = 5` this is 6 + 4 + 1 = 11. Replace every place in the parent code that uses `context_size + 2` with this value (positional encoding length, masks, average pooling weights).
4. **Type ids.** A registered, non-persistent buffer: 0 for observation tokens, 1 for traversability tokens, 2 for the goal.
5. **Feature extraction.** The parent calls `extract_features` on the observation encoder. Replace it with `extract_endpoints`:
   - take the last endpoint, pool it, and use it for the observation tokens (as before)
   - the frames are stacked `(T*B, ...)` in time order, so the current frame is the last `B` entries; take the stride 8, 16, 32 endpoints from those and send them to the decoder
6. **Trav tokens.** Run decoder, then tokenizer. During training, with probability `trav_dropout` per sample, set that sample's trav tokens to zero.
7. **Token order.** `[6 obs, 4 trav, 1 goal]`. The goal must stay last so the existing goal-mask logic (last index) still works.
8. **After positional encoding:** add the type embedding.
9. **Store the map.** Set `self.last_trav_map` so the training loop can read it. Do not change the return value, so deployment code keeps working.
10. **`use_trav = False` path.** Skip steps 5 (decoder part), 6 and 7, and use the parent's token layout. This is what Baseline B uses.

Pretrained weight loading (put it in a helper function in this file or in `train.py`):

- Load `nomad.pth` with `strict=False`.
- Print missing and unexpected keys.
- Expected missing: decoder, tokenizer, type embedding.
- Check whether the positional encoding has parameters. If NoMaD's is sinusoidal with no parameters, nothing to re-init.
- If the checkpoint's stored positional encoding length differs, it will show up as an unexpected size mismatch; handle that explicitly.

### C2. Shape tests for the encoder

Add to `tests/` a test that, with `B = 2` and random inputs:

- token sequence is `(2, 11, 256)` before pooling
- pooled context is `(2, 256)`
- runs with the goal mask at 0 and at 1
- `last_trav_map` is `(2, 1, 64, 64)`
- the diffusion head accepts the context unchanged

### C3. `train/vint_train/data/vint_dataset.py` (modify)

1. **New arguments.** Add `teacher_dir` and `teacher_mode` to the constructor and store them.
2. **`load_teacher(traj_name, t)`.** Builds the path `<teacher_dir>/<dataset_name>/<traj_name>/<t>.npy`, loads it, converts to float32, and adds a channel dimension: `(1, 64, 64)`.
3. **`__getitem__`.** Find the variable that holds the trajectory name and the index of the current observation (upstream: `f_curr`, `curr_time`). Call `load_teacher` for that index and append the result to the returned tuple.
4. **Path mode.** If `teacher_mode == "path"`, return a zeros `(1, 64, 64)` tensor so the batch structure is the same.
5. **Missing file.** Decide on a policy and implement it once: raise an error (safest at first) or return zeros plus a valid flag.
6. **Wherever the dataset is built** (`train.py`), pass the new arguments from the config.

Visual check, before any training: pick 20 dataset items, plot the returned observation frame next to the returned teacher map, and confirm they match.

---

## Part D. Training

### D1. `train/vint_train/training/train_utils.py` (modify)

1. Copy `train_nomad` to `train_travnav` and `evaluate_nomad` to `evaluate_travnav`.
2. Update the batch unpacking to include the teacher map and move it to the device.
3. After the forward pass, read `encoder.last_trav_map` (remember `.module` if the model is wrapped in DataParallel).
4. Compute the target with `get_teacher_map`, passing the raw waypoint tensor before delta computation and normalisation.
5. Compute `l_distil` with `distil_loss`.
6. Total loss: `alpha * dist_loss + (1 - alpha) * diffusion_loss + lambda_t * l_distil`.
7. Log the three losses separately.
8. Every N steps save a PNG with current frame, predicted map, teacher map.
9. In `evaluate_travnav`, also compute the mean L1 between predicted and teacher maps on the test split. This is your map-quality number for the report.

Stage handling:

- `stage: 1`: set `requires_grad = False` for everything except the decoder, tokenizer and type embedding. Use only `l_distil` as the loss.
- `stage: 2`: train everything with the combined loss, with two parameter groups: new modules at `1e-4`, pretrained modules at `1e-5`.

### D2. `train/train.py` (modify)

- Register `model_type: travnav` in the model-construction branch.
- Build the model with the new constructor arguments from the config.
- Call the pretrained weight loading helper.
- Choose `train_travnav` / `evaluate_travnav` when `model_type` is `travnav`.
- Build the optimizer with the two parameter groups in stage 2.
- Support `load_run` so stage 2 starts from the best stage 1 checkpoint.

### D3. Configs (new)

Create from `nomad_ft_baseline.yaml`:

- `train/config/travnav_stage1.yaml`
- `train/config/travnav_stage2.yaml`

Keys to add or change:

| Key | Value |
|---|---|
| `model_type` | `travnav` |
| `context_size` | `5` |
| `num_trav_tokens` | `4` |
| `trav_dropout` | `0.1` |
| `use_trav` | `true` |
| `teacher` | `dino` |
| `teacher_dir` | `/data/teacher_maps` |
| `lambda_t` | `0.1` (first guess) |
| `stage` | `1` or `2` |
| `num_diffusion_iters` | `10` |
| `batch_size` | `64` |

Make a copy with `use_trav: false` for Baseline B, keeping seed, epochs, batch size, schedule and data identical.

### D4. Overfit test

Before any real run: train on a single batch for 300 steps.

- `l_distil` should drop clearly.
- `l_diff` should drop clearly.
- Predicted map should start to look like the teacher map.

If it does not, fix it now. Do not start stage 1 on the full data.

---

## Order of work

| Step | What | Depends on |
|---|---|---|
| 1 | Modify `make_teacher_maps.py` (A1) | nothing |
| 2 | Run with `--limit 16`, time it | 1 |
| 3 | Write and run `check_teacher_maps.py` (A2) | 2 |
| 4 | Fix teacher if needed, repeat 2 and 3 | 3 |
| 5 | Start full generation in background (A3) | 4 |
| 6 | Write `trav_modules.py` and its tests (B) | Day 1 channel numbers |
| 7 | Write `travnav_vint.py` and shape tests (C1, C2) | 6 |
| 8 | Modify the dataset (C3) and check frame/map alignment | 5 (at least partly) |
| 9 | Modify `train_utils.py` and `train.py`, create configs (D) | 7, 8 |
| 10 | Overfit test (D4) | 9 |
| 11 | Stage 1 run, then stage 2 run, then Baseline B | 10, full maps done |

Steps 6 and 7 do not need the maps, so they run while step 5 is generating.

## Common mistakes to watch for

- Map file index not equal to the dataset's current frame index. Check visually on 20 items.
- Student map collapses to all zeros. Check the teacher maps are not mostly zero.
- Seed region on a person or wall producing confident wrong maps. Use the seed spread ranking from A2.
- Using `context_size + 2` somewhere in the parent code that you forgot to replace.
- Goal token no longer last, which breaks the goal mask.
- Comparing TravNav only against zero-shot NoMaD. The fair comparison is Baseline B.