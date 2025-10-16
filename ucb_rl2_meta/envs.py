import gym
import numpy as np
import torch
from PIL import Image
from gym.spaces.box import Box
from scipy.ndimage import gaussian_filter

from baselines.common.vec_env.vec_env import VecEnvWrapper


def fast_perceptual_downscale(
        img_arr: np.ndarray,
        factor: float = 2.0,
        # Resampling choice: BOX is very fast, LANCZOS is sharper but slower
        resample=Image.Resampling.BOX,
        # Preprocessing
        chroma_sigma_base: float = 0.15,  # small smoothing to reduce color aliasing
        luma_sigma_base: float = 0.25,  # tiny blur to stabilize gradients
        # Edge-aware boosts
        luma_boost_strength: float = 28.0,  # boost luminance near edges
        chroma_boost_strength: float = 16.0,  # boost chroma near edges for color crispness
        # Post-sharpen
        unsharp_amount: float = 0.45,
        unsharp_sigma: float = 0.6
) -> np.ndarray:
    if factor <= 0:
        raise ValueError("factor must be > 0")

    h, w = img_arr.shape[:2]
    dst_w = max(1, int(round(w / factor)))
    dst_h = max(1, int(round(h / factor)))

    # Convert to PIL for fast resize and YCbCr conversion
    src = Image.fromarray(img_arr)
    ycbcr = src.convert("YCbCr")
    y, cb, cr = [np.array(c, dtype=np.float32) for c in ycbcr.split()]

    # Lightweight pre-blur (scale-aware but cheap)
    scale = max(1.0, factor)
    luma_sigma = luma_sigma_base * scale * 0.4
    chroma_sigma = chroma_sigma_base * scale * 0.4
    if luma_sigma > 0:
        y_blur = gaussian_filter(y, sigma=luma_sigma)
    else:
        y_blur = y
    if chroma_sigma > 0:
        cb_s = gaussian_filter(cb, sigma=chroma_sigma)
        cr_s = gaussian_filter(cr, sigma=chroma_sigma)
    else:
        cb_s, cr_s = cb, cr

    # Fast gradient magnitude (cheaper than Sobel)
    # Horizontal and vertical diffs
    gx = np.diff(y_blur, axis=1, append=y_blur[:, -1:])
    gy = np.diff(y_blur, axis=0, append=y_blur[-1:, :])
    edge = np.sqrt(gx * gx + gy * gy)
    edge_norm = edge / (edge.max() + 1e-6)

    # Edge-aware boosts: luma and chroma
    y_boost = np.clip(y_blur + luma_boost_strength * edge_norm, 0, 255)
    cb_boost = np.clip(cb_s + chroma_boost_strength * edge_norm, 0, 255)
    cr_boost = np.clip(cr_s + chroma_boost_strength * edge_norm, 0, 255)

    boosted = Image.merge(
        "YCbCr",
        [
            Image.fromarray(y_boost.astype(np.uint8)),
            Image.fromarray(cb_boost.astype(np.uint8)),
            Image.fromarray(cr_boost.astype(np.uint8)),
        ],
    ).convert("RGB")

    # Fast downscale
    down = boosted.resize((dst_w, dst_h), resample=resample)

    # Mild unsharp mask on luma after downscale to restore micro-contrast
    y2, cb2, cr2 = [np.array(c, dtype=np.float32) for c in down.convert("YCbCr").split()]
    y2_blur = gaussian_filter(y2, sigma=unsharp_sigma)
    y2_sharp = np.clip(y2 + unsharp_amount * (y2 - y2_blur), 0, 255).astype(np.uint8)

    out = Image.merge(
        "YCbCr",
        [
            Image.fromarray(y2_sharp),
            Image.fromarray(cb2.astype(np.uint8)),
            Image.fromarray(cr2.astype(np.uint8)),
        ],
    ).convert("RGB")

    return np.array(out)


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImageProcgen(TransposeObs):
    def __init__(self, env=None, op=[0, 3, 2, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImageProcgen, self).__init__(env)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[2], obs_shape[1], obs_shape[0]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        if ob.shape[0] == 1:
            ob = ob[0]
        return ob.transpose(self.op[0], self.op[1], self.op[2], self.op[3])


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecPyTorchProcgenSmall(VecPyTorchProcgen):
    def __init__(self, venv, device):
        super().__init__(venv, device)  # correct super
        self.device = device
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32
        )

    def _downscale_batch(self, obs_np):
        b, h, w, c = obs_np.shape
        if h == 32 and w == 32:
            return obs_np
        dst_h, dst_w = 32, 32
        out = np.empty((b, dst_h, dst_w, c), dtype=np.uint8)
        for i in range(b):
            img = Image.fromarray(obs_np[i], mode="RGB")
            down = img.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
            out[i] = np.array(down, dtype=np.uint8)
        return out

    def reset(self):
        obs = self.venv.reset()
        if obs.ndim == 4 and obs.shape[-1] == 3:
            obs = self._downscale_batch(obs)
            obs = obs.transpose(0, 3, 1, 2)
        elif obs.ndim == 4 and obs.shape[1] == 3:
            obs = obs.transpose(0, 2, 3, 1)
            obs = self._downscale_batch(obs)
            obs = obs.transpose(0, 3, 1, 2)
        else:
            raise ValueError("Unexpected observation shape")
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.ndim == 4 and obs.shape[-1] == 3:
            obs = self._downscale_batch(obs)
            obs = obs.transpose(0, 3, 1, 2)
        elif obs.ndim == 4 and obs.shape[1] == 3:
            obs = obs.transpose(0, 2, 3, 1)
            obs = self._downscale_batch(obs)
            obs = obs.transpose(0, 3, 1, 2)
        else:
            raise ValueError("Unexpected observation shape")
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
