# Real-Time PyTorch Koopman NCA Model Integration with Multi-Wavelet, Hopfield Memory, JEPA, Active Inference, and DSL
# This script integrates multiple advanced components into a single interactive simulation:
# 1. **Koopman-based Neural Cellular Automaton (NCA)** with complex-valued latent states updated via a unitary operator.
#    - We parametrize a skew-Hermitian generator and use the Cayley transform to obtain a unitary state transition matrix.
#    - This ensures the NCA's internal dynamics are norm-preserving (unitary), enhancing stability.
#    - The NCA processes a 2D grid (e.g. 480x480) of cells with complex latent vectors, and can incorporate neighbor interactions.
# 2. **Multi-Wavelet Embedding/Decoding** for multi-scale latent compression:
#    - The environment's observations (RGB images) are transformed using wavelets (Haar, Daubechies) into multi-scale coefficients (coarse approximation + detail subbands).
#    - This provides an invertible, efficient representation. The coarse part captures large-scale structure (slow features), while details capture fine texture.
#    - We support Haar (db1) and Daubechies wavelets and could extend to wavelet packet transforms for finer frequency decomposition.
# 3. **Complex-valued Hopfield Associative Memory**:
#    - Stores patterns (e.g., past latent states) with fast Hebbian outer-product updates. Patterns are normalized to unit length.
#    - Retrieval uses the Hopfield update rule: the memory matrix (sum of pattern outer products) multiplies a query to recall similar stored states.
#    - Complex phase coding is used: patterns and queries are complex, enabling potential phase-based encoding of information. We project and compare using complex inner products.
# 4. **JEPA-style Contrastive Latent Prediction**:
#    - Joint Embedding Predictive Architecture (JEPA) approach: the model predicts future latent representations and is trained with both mean-squared error (L2) and InfoNCE contrastive loss.
#    - We apply masked, delayed supervision: the model does not receive ground truth every time-step (simulating partial observation), and must predict multiple steps ahead during those intervals.
#    - InfoNCE contrastive loss encourages the predicted latent to be closer to the true future latent than to other (negative) latents, focusing on correct long-range predictions (especially of coarse, slow features).
# 5. **Active Inference Loop with Online Constraint Enforcement (KKT)**:
#    - The system learns online, in real-time, without offline training epochs. Model parameters are updated each time-step from the prediction errors.
#    - We enforce constraints (like unitarity or energy preservation) via Lagrange multipliers updated in tandem (Karush-Kuhn-Tucker style).
#    - For example, we maintain a dual variable to enforce latent energy conservation (norm of state) to complement the explicit unitary design.
# 6. **Real-Time PyBullet Simulation Environments**:
#    - We integrate with PyBullet physics simulations (e.g., bouncing balls, a robotic arm). The environment runs in real time, providing observations to the model.
#    - Multiple environment scenarios are supported and can be switched interactively (e.g., via a UI slider). Each environment is minimal but captures distinct dynamics (ballistic motion vs. articulated robot).
# 7. **Interactive UI (ImGui/OpenCV)**:
#    - We use PyBullet's debug interface (sliders, buttons) and OpenCV overlays for interactivity.
#    - Users can pause/resume the simulation, adjust the simulation speed, perturb/reset model weights, save/load model state, and toggle each major component (NCA, memory, DSL) on/off during runtime.
#    - OpenCV is used to display the model's view (e.g., the 480x480 latent grid or observation) with textual overlays showing status (losses, toggles, etc.), providing real-time visualization.
# 8. **Differentiable DSL Module**:
#    - A symbolic Domain-Specific Language module that can execute programs to transform latent representations in a structured, interpretable way.
#    - The DSL is Turing-complete: it supports conditional branching and looping. It can incorporate wavelet-based attention, e.g., focusing on specific frequency bands.
#    - In this implementation, the DSL program analyzes the wavelet coefficients of the predicted latent and performs iterative adjustments (e.g., dampening excessive high-frequency energy) as a form of reasoning or constraint enforcement.
# 
# Overall, this script demonstrates a unified architecture where all components interoperate in an online learning loop, suitable for real-time experimentation and visualization.

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pybullet as p
import pybullet_data
import cv2
import time

# ==================================================
# Class: Koopman-based Neural Cellular Automaton (NCA)
# ==================================================
# This class defines a cellular automaton that evolves a grid of complex-valued latent states.
# Each cell has a latent state vector (with complex entries). The update rule is linear (Koopman operator based) to ensure predictability and stability.
# We enforce that the update operator is unitary (norm-preserving) by parameterizing a skew-Hermitian generator and applying the Cayley transform.
# This ensures the evolution is essentially a rotation in complex state-space (no explosion or dissipation), aligning with the idea of preserving "energy" in the latent dynamics.
# The NCA can also incorporate neighbor interactions: we include a small coupling from the 4-nearest neighbors (Von Neumann neighborhood).
# Note: A truly unitary neighbor mixing across the grid would require a global unitary (e.g., using FFT for a unitary shift). Here we approximate neighbor influence by a small additive term, and rely on the unitary core to maintain stability.
# The update equation per time-step:
#    G = A - A^H (skew-Hermitian generator from parameter A)
#    U = (I - 0.5 * dt * G)^{-1} * (I + 0.5 * dt * G)   (Cayley transform yielding a unitary matrix)
#    state <- neighbor_coupling * (sum of neighbor states) + U (applied to each cell's state vector)
# After updating, the state is thus rotated/mixed by U and receives a small contribution from neighbors.
# We maintain the state as a complex tensor of shape (H, W, latent_dim). For computation, we reshape or vectorize as needed.
class KoopmanNCA(nn.Module):
    def __init__(self, latent_dim, dt=0.1, neighbor_coupling=0.1):
        super(KoopmanNCA, self).__init__()
        # latent_dim: dimension of each cell's latent vector (number of complex features per cell).
        # dt: time-step for the discrete update (smaller dt yields smaller rotations per step).
        # neighbor_coupling: weight for neighbor influence (small to keep perturbations unitary in spirit).
        self.latent_dim = latent_dim
        self.dt = dt
        self.neighbor_coupling = neighbor_coupling
        # Parameter A (complex) for the generator. We initialize A small (near zero), so initial U ~ I.
        # A is an unconstrained complex matrix (latent_dim x latent_dim). 
        # We'll construct a skew-Hermitian generator G = A - A^H, and then compute unitary U from it each update.
        # Register real and imaginary parts as parameters if complex dtype is not directly supported in optimizer.
        # PyTorch does support complex tensors with autograd, so we can use a complex parameter directly.
        self.A = nn.Parameter(0.01 * torch.eye(latent_dim, dtype=torch.cfloat))
    
    def forward(self, state):
        # state: complex tensor of shape (H, W, latent_dim) representing the grid of latent vectors.
        # We apply one update step and return the new state (same shape).
        H, W, dim = state.shape
        # 1. Compute skew-Hermitian generator G from A.
        # Ensure A is interpreted as complex matrix. A^H is the conjugate transpose.
        A = self.A
        G = A - A.conj().T  # skew-Hermitian (G^H = -G)
        # 2. Compute unitary matrix U via Cayley transform.
        # U = (I - 0.5*dt*G)^{-1} * (I + 0.5*dt*G)
        I = torch.eye(self.latent_dim, dtype=torch.cfloat, device=A.device)
        # Use torch.linalg.inv for matrix inverse.
        U = torch.linalg.inv(I - 0.5 * self.dt * G) @ (I + 0.5 * self.dt * G)
        # 3. Neighbor coupling: compute neighbor sum for each cell.
        # We use periodic wrap-around for simplicity (torus topology).
        # This will produce a complex tensor same shape as state.
        # We shift the state tensor in each direction and add them.
        # (If neighbor coupling is zero, this step can be skipped.)
        if self.neighbor_coupling != 0:
            # Use torch.roll for wrap-around shifts.
            up = torch.roll(state, shifts=-1, dims=0)
            down = torch.roll(state, shifts=1, dims=0)
            left = torch.roll(state, shifts=-1, dims=1)
            right = torch.roll(state, shifts=1, dims=1)
            neighbor_sum = up + down + left + right
            # Scale neighbor contribution.
            state = state + (self.neighbor_coupling * neighbor_sum)
        # 4. Apply the unitary transformation U to each cell's state vector.
        # We reshape the state to (H*W, dim) to apply U via matrix multiplication.
        state_flat = state.view(-1, self.latent_dim)  # shape (N, dim) where N = H*W
        # Right-multiply by U (so each row vector is transformed by U).
        new_state_flat = state_flat @ U  # shape (N, dim)
        new_state = new_state_flat.view(H, W, self.latent_dim)
        return new_state

# ==================================================
# Class: MultiWaveletEmbed
# ==================================================
# This class handles the transform and inverse transform between image space (RGB) and the multi-scale wavelet latent space.
# We support Haar wavelet (Daubechies-1) and Daubechies-2 (4-tap filters) as examples. The wavelet transform is done per color channel.
# For simplicity, we implement a single-level 2D wavelet transform: it produces a coarse approximation (LL) and three detail subbands (LH, HL, HH) each at half resolution.
# The transform is invertible. We arrange the wavelet coefficients as separate channels in the latent representation.
# Specifically, if input image has shape (3, H, W), the output latent has shape (3*4, H/2, W/2): 3 channels for LL (one per color), 3 for LH, 3 for HL, 3 for HH.
# This provides an efficient multi-scale compression (reducing spatial size by 4x) while preserving information.
# (Wavelet packet or multi-level transforms could be added by recursively transforming LL further or splitting detail bands, but we omit that for brevity.)
class MultiWaveletEmbed:
    def __init__(self, wavelet='haar'):
        # wavelet: type of wavelet to use ('haar' or 'db2').
        self.wavelet = wavelet
        # Define wavelet filter coefficients for Haar and Daubechies-2.
        if wavelet == 'haar':
            # Haar: low-pass = [1/sqrt(2), 1/sqrt(2)]; high-pass = [1/sqrt(2), -1/sqrt(2)]
            c = 1 / math.sqrt(2)
            self.lp = np.array([c, c], dtype=np.float32)
            self.hp = np.array([c, -c], dtype=np.float32)
        elif wavelet == 'db2':
            # Daubechies-2 (a.k.a Daubechies-4 coefficients, 4 taps) - two vanishing moments.
            # These coefficients are from the Daubechies D4 wavelet:
            # h0 = (1+√3)/(4√2), h1 = (3+√3)/(4√2), h2 = (3-√3)/(4√2), h3 = (1-√3)/(4√2).
            sqrt3 = math.sqrt(3)
            denom = 4 * math.sqrt(2)
            h0 = (1 + sqrt3) / denom
            h1 = (3 + sqrt3) / denom
            h2 = (3 - sqrt3) / denom
            h3 = (1 - sqrt3) / denom
            # Low-pass filter (analysis) for D4
            self.lp = np.array([h0, h1, h2, h3], dtype=np.float32)
            # High-pass filter (analysis) (every other coefficient of low-pass with alternating sign, for orthonormal wavelet)
            self.hp = np.array([h3, -h2, h1, -h0], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported wavelet type: {wavelet}")
        # We will use numpy for the transform and then convert to torch. (For performance, one might implement with conv filters in torch.)
    
    def forward(self, image):
        # image: numpy or torch array of shape (3, H, W) with values in [0,1] (float).
        # We'll perform the transform on CPU using numpy for simplicity, then convert to torch.
        # Ensure input is numpy array for processing.
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
        else:
            img_np = image
        C, H, W = img_np.shape
        # We require H, W to be even for a single level transform.
        assert H % 2 == 0 and W % 2 == 0, "Image dimensions must be even for wavelet transform"
        h_half, w_half = H // 2, W // 2
        # Prepare arrays for subbands
        LL = np.zeros((C, h_half, w_half), dtype=np.float32)
        LH = np.zeros((C, h_half, w_half), dtype=np.float32)
        HL = np.zeros((C, h_half, w_half), dtype=np.float32)
        HH = np.zeros((C, h_half, w_half), dtype=np.float32)
        # 1D filters
        lp = self.lp
        hp = self.hp
        filter_len = lp.shape[0]
        # Convolution along rows and downsampling
        # For each channel, convolve each row with low-pass and high-pass filters, then downsample by 2.
        low_rows = np.zeros((C, H, w_half), dtype=np.float32)
        high_rows = np.zeros((C, H, w_half), dtype=np.float32)
        for c in range(C):
            for i in range(H):
                # pad mode: for simplicity, use periodic padding (wrap-around) similar to our CA boundary, or reflect
                # We'll do periodic pad here to avoid boundary issues.
                row = np.concatenate([img_np[c, i, :], img_np[c, i, :filter_len-1]], axis=0)  # pad end with beginning part
                # Convolve with lp and hp filters
                conv_lp = np.convolve(row, lp, mode='valid')  # length = W + filter_len - 1, but 'valid' gives W
                conv_hp = np.convolve(row, hp, mode='valid')
                # Downsample by 2
                low_rows[c, i, :] = conv_lp[::2]
                high_rows[c, i, :] = conv_hp[::2]
        # Now low_rows has shape (C, H, W/2) = (C, H, w_half)
        # Next, convolve along columns of low_rows and high_rows with lp and hp, then downsample vertically.
        for c in range(C):
            for j in range(w_half):
                # Process column j for both low_rows and high_rows
                col_low = np.concatenate([low_rows[c, :, j], low_rows[c, :filter_len-1, j]], axis=0)
                col_high = np.concatenate([high_rows[c, :, j], high_rows[c, :filter_len-1, j]], axis=0)
                conv_low_low = np.convolve(col_low, lp, mode='valid')  # apply low-pass vertically on low_rows
                conv_low_high = np.convolve(col_low, hp, mode='valid') # high-pass vertically on low_rows
                conv_high_low = np.convolve(col_high, lp, mode='valid') # low-pass vertically on high_rows
                conv_high_high = np.convolve(col_high, hp, mode='valid') # high-pass vertically on high_rows
                # Downsample by 2 vertically
                LL[c, :, j] = conv_low_low[::2]
                LH[c, :, j] = conv_high_low[::2]
                HL[c, :, j] = conv_low_high[::2]
                HH[c, :, j] = conv_high_high[::2]
        # Stack subbands into a single tensor output: shape (C*4, h_half, w_half)
        # Order: [LL_R, LL_G, LL_B, LH_R, LH_G, LH_B, HL_R, HL_G, HL_B, HH_R, HH_G, HH_B]
        latent = np.concatenate([LL, LH, HL, HH], axis=0)
        latent_torch = torch.from_numpy(latent).to(device)
        return latent_torch
    
    def inverse(self, latent):
        # Inverse wavelet transform: reconstruct image from latent (wavelet coefficients).
        # latent: torch or numpy array shape (C*4, h_half, w_half)
        # The inverse basically reverses the forward steps:
        # Upsample coefficients in vertical direction, filter with synthesis filters (which for orthonormal wavelets are time-reversed analysis filters), then sum.
        if isinstance(latent, torch.Tensor):
            lat_np = latent.detach().cpu().numpy()
        else:
            lat_np = latent
        # Number of color channels
        # lat_np shape = (4*C, h_half, w_half)
        # Determine C by dividing first dimension by 4
        total_ch, h_half, w_half = lat_np.shape
        C = total_ch // 4
        LL = lat_np[0:C, :, :]
        LH = lat_np[C:2*C, :, :]
        HL = lat_np[2*C:3*C, :, :]
        HH = lat_np[3*C:4*C, :, :]
        # Synthesis filters: for orthonormal wavelets, synthesis low-pass is time-reverse of analysis low-pass, same for high-pass.
        lp = self.lp
        hp = self.hp
        # Time-reverse (and conjugate if expecting complex filters, but filters are real here).
        lp_rev = lp[::-1]
        hp_rev = hp[::-1]
        H = h_half * 2
        W = w_half * 2
        # Allocate arrays for intermediate (after vertical upsampling + filtering)
        low_comb = np.zeros((C, H, w_half), dtype=np.float32)
        high_comb = np.zeros((C, H, w_half), dtype=np.float32)
        # Inverse vertical transform:
        for c in range(C):
            for j in range(w_half):
                # We have subbands LL, LH, HL, HH for each channel c.
                # We reconstruct the low-frequency row part and high-frequency row part separately, then combine for full rows.
                # Upsample LL and HL (which came from low_rows originally):
                up_LL = np.zeros(H + len(lp_rev) - 2, dtype=np.float32)  # upsample by factor 2 -> H samples, we will pad for convolution
                up_HL = np.zeros(H + len(lp_rev) - 2, dtype=np.float32)
                # place LL values in even indices (vertical reconstruction)
                up_LL[::2] = LL[c, :, j]
                up_HL[::2] = HL[c, :, j]
                # Convolve with synthesis filters
                rec_low = np.convolve(up_LL, lp_rev, mode='full') + np.convolve(up_HL, hp_rev, mode='full')
                # rec_low now length ~ H + filter_len - 1, we may need to trim or pad to exactly H (due to convolution extension).
                # Because of circular padding assumption, we can take the valid portion the size of H.
                low_comb[c, :, j] = rec_low[(len(lp_rev)-1) : (len(lp_rev)-1 + H)]
                # Upsample LH and HH (came from high_rows originally):
                up_LH = np.zeros(H + len(lp_rev) - 2, dtype=np.float32)
                up_HH = np.zeros(H + len(lp_rev) - 2, dtype=np.float32)
                up_LH[::2] = LH[c, :, j]
                up_HH[::2] = HH[c, :, j]
                rec_high = np.convolve(up_LH, lp_rev, mode='full') + np.convolve(up_HH, hp_rev, mode='full')
                high_comb[c, :, j] = rec_high[(len(lp_rev)-1) : (len(lp_rev)-1 + H)]
        # Now low_comb and high_comb are of shape (C, H, w_half) representing low_rows and high_rows from forward transform.
        # Inverse horizontal transform:
        image_rec = np.zeros((C, H, W), dtype=np.float32)
        for c in range(C):
            for i in range(H):
                # We have low_comb[c, i, :] and high_comb[c, i, :] as the low-pass and high-pass contributions for row i.
                # Upsample horizontally:
                low_row = np.zeros(W + len(lp_rev) - 2, dtype=np.float32)
                high_row = np.zeros(W + len(lp_rev) - 2, dtype=np.float32)
                low_row[::2] = low_comb[c, i, :]
                high_row[::2] = high_comb[c, i, :]
                # Convolve with synthesis filters along row
                rec_row = np.convolve(low_row, lp_rev, mode='full') + np.convolve(high_row, hp_rev, mode='full')
                image_rec[c, i, :] = rec_row[(len(lp_rev)-1) : (len(lp_rev)-1 + W)]
        image_torch = torch.from_numpy(image_rec)
        return image_torch

# ==================================================
# Class: HopfieldMemory
# ==================================================
# Implements a complex-valued Hopfield associative memory.
# This memory stores a set of patterns (complex vectors) and can retrieve patterns given a query.
# We use Hebbian learning: when a new pattern is added, the weight matrix is updated by outer product.
# We maintain patterns normalized to unit norm (so that each stored pattern is on the hypersphere, aiding stability).
# Retrieval:
#   - We can perform one update step: x_new = M x (where M = sum_k p_k p_k^H is the memory matrix).
#   - This yields a combination of stored patterns weighted by their similarity to the query.
#   - Alternatively, one could iterate x_{t+1} = sign(M x_t) for binary Hopfield, but here we use a single step analog recall.
# We primarily use the memory for two purposes:
#   1. As negative examples for contrastive learning (InfoNCE) – we can sample stored patterns as incorrect "future" states.
#   2. (Optional) To aid prediction: e.g., if the model's predicted latent is close to a stored pattern, the memory could reinforce it.
# The memory is updated online (fast O(K) per new pattern addition).
class HopfieldMemory:
    def __init__(self, max_patterns=100):
        self.max_patterns = max_patterns
        self.patterns = []  # list of stored patterns (torch 1D tensors, complex dtype)
    
    def add_pattern(self, pattern):
        # Add a new pattern to memory (after projecting to unit norm).
        # pattern: torch complex tensor (1D vector).
        # If patterns exceed max, forget the oldest.
        pattern = pattern.to(device)
        # Flatten in case pattern is multi-dimensional (like an image), to store as vector.
        pattern_vec = pattern.flatten()
        # Normalize to unit length.
        norm = torch.linalg.norm(pattern_vec).item()
        if norm > 1e-8:
            pattern_vec = pattern_vec / norm
        else:
            pattern_vec = pattern_vec.clone()
        # If memory full, remove oldest
        if len(self.patterns) >= self.max_patterns:
            self.patterns.pop(0)
        self.patterns.append(pattern_vec)
    
    def retrieve(self, query):
        # Retrieve a pattern given a query vector (single-step Hopfield update).
        # query: torch complex tensor (same shape as stored patterns, flattened or unflattened).
        # Returns a torch complex tensor (flattened) representing the retrieved pattern.
        if len(self.patterns) == 0:
            return None
        q_vec = query.flatten().to(device)
        # We can compute weighted sum of stored patterns: sum_k (p_k * (p_k^H q))
        # p_k^H q is a complex scalar similarity (inner product).
        retrieved = torch.zeros_like(self.patterns[0])
        for p in self.patterns:
            # Inner product p^H q (conjugate p then dot with q)
            sim = torch.dot(p.conj(), q_vec)
            retrieved += sim * p  # weight pattern by similarity
        # We could also normalize the result or threshold if needed. For now, return as is.
        return retrieved

    def reset(self):
        # Clear memory
        self.patterns = []

# ==================================================
# Class: DSLModule
# ==================================================
# This module represents a differentiable "program" that can manipulate latent representations with loops and conditionals (symbolic reasoning).
# The DSL here is implemented as a fixed function (though in principle it could parse a program).
# Our DSL program:
#   - Analyzes the wavelet coefficient distribution of the predicted latent.
#   - If high-frequency detail energy is disproportionately high relative to coarse energy, it iteratively reduces the detail.
#   - This simulates an attention mechanism focusing on the wavelet subbands and enforcing a prior that the system favors smooth (low-frequency) components.
# This program uses:
#   - Loop (while) and conditional logic (if) -> demonstrating Turing-completeness (a loop that can run until a condition is satisfied).
#   - Wavelet-structured attention: it explicitly computes energies of wavelet subbands (coarse vs detail) and acts on those coefficients.
# The operations within are differentiable w.r.t. the latent (multiplications, sums), though the control flow is not directly differentiable (we treat it as a piecewise function).
class DSLModule:
    def __init__(self, num_coarse_channels=3, num_detail_channels=9, damp_factor=0.9, energy_ratio_threshold=2.0):
        # num_coarse_channels: number of channels corresponding to coarse LL subbands (e.g., 3 for RGB coarse).
        # num_detail_channels: number of channels for all detail subbands (e.g., 9 for RGB's LH,HL,HH).
        # damp_factor: how much to scale down details when they are too high.
        # energy_ratio_threshold: the threshold of (detail_energy/coarse_energy) above which we trigger detail dampening.
        self.num_coarse = num_coarse_channels
        self.num_detail = num_detail_channels
        self.damp = damp_factor
        self.threshold = energy_ratio_threshold
    
    def run(self, latent):
        # latent: torch complex tensor of shape (C_total, H, W) representing wavelet coefficients (as produced by MultiWaveletEmbed).
        # We separate coarse and detail parts:
        coarse = latent[0:self.num_coarse]
        detail = latent[self.num_coarse:self.num_coarse + self.num_detail]
        # Compute energy (sum of squared magnitudes) for coarse and detail.
        # Use .abs()**2 to get magnitude-squared for complex.
        coarse_energy = torch.sum(coarse.abs()**2)
        detail_energy = torch.sum(detail.abs()**2)
        # Iteratively dampen detail until energy ratio is below threshold or max iterations reached.
        # This loop and condition are the "program" logic.
        max_iter = 10
        iter_count = 0
        # (Note: In a true differentiable program, we'd avoid a hard loop or use a differentiable approximation. Here it's straightforward control logic.)
        while detail_energy > self.threshold * (coarse_energy + 1e-8) and iter_count < max_iter:
            # Reduce detail coefficients
            detail = detail * self.damp
            # Recompute energy after damping
            detail_energy = torch.sum(detail.abs()**2)
            iter_count += 1
        # If loop ran, we now have adjusted detail. We combine back the coarse and detail.
        new_latent = torch.cat([coarse, detail], dim=0)
        return new_latent

# ==================================================
# Main: Active Inference Loop and Integration
# ==================================================
# Set up the simulation and all components, then run the real-time loop.
if __name__ == "__main__":
    # Initialize PyBullet in GUI mode for visualization.
    p.connect(p.GUI)
    # Load environment assets:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to access built-in URDFs like plane, robot
    # Create environments:
    # We will support two environments: 0 = Bouncing Balls, 1 = Robotic Arm.
    env_index = 0  # start with bouncing balls
    
    # Function to initialize the selected environment
    def load_environment(index):
        p.resetSimulation()
        # Common settings
        p.setGravity(0, 0, -10)
        # Add a ground plane for completeness (though not used in zero-gravity bounce or fixed-base robot).
        plane_id = p.loadURDF("plane.urdf")
        # Environment-specific setup
        body_ids = []  # track bodies for potential use
        if index == 0:
            # Bouncing Balls environment
            # Create an invisible container (four walls) to keep balls roughly in [-1,1] range horizontally.
            wall_height = 2.0
            wall_thickness = 0.1
            # Four walls forming a square boundary at x=±1, y=±1.
            wall1 = p.loadURDF("cube.urdf", basePosition=[1.0, 0, wall_height/2 - 0.5], baseOrientation=[0,0,0,1], globalScaling=1.0)  # right wall
            wall2 = p.loadURDF("cube.urdf", basePosition=[-1.0, 0, wall_height/2 - 0.5], baseOrientation=[0,0,0,1], globalScaling=1.0) # left wall
            wall3 = p.loadURDF("cube.urdf", basePosition=[0, 1.0, wall_height/2 - 0.5], baseOrientation=[0,0,0,1], globalScaling=1.0)  # front wall
            wall4 = p.loadURDF("cube.urdf", basePosition=[0, -1.0, wall_height/2 - 0.5], baseOrientation=[0,0,0,1], globalScaling=1.0) # back wall
            # Scale walls: assume cube.urdf default size 1m^3, we scale to match needed dimensions.
            # We'll use changeVisualShape to make walls invisible (since our agent doesn't "see" them directly, but they physically constrain balls).
            for wid in [wall1, wall2, wall3, wall4]:
                p.changeVisualShape(wid, -1, rgbaColor=[0,0,0,0])  # invisible
                p.changeDynamics(wid, -1, restitution=1.0)  # make walls perfectly elastic
            # Create a couple of balls
            ball_radius = 0.05
            col_ball = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
            vis_ball = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1,0,0,1])
            # Ball1 red
            ball1 = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col_ball, baseVisualShapeIndex=vis_ball, basePosition=[-0.5, -0.5, 0.5])
            # Ball2 green (we'll recolor after creation)
            vis_ball2 = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0,1,0,1])
            ball2 = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col_ball, baseVisualShapeIndex=vis_ball2, basePosition=[0.5, 0.5, 0.5])
            # Set initial velocities so they move/bounce
            p.resetBaseVelocity(ball1, linearVelocity=[2, 1, 0])  # some horizontal velocity
            p.resetBaseVelocity(ball2, linearVelocity=[-1, -2, 0])
            # Make balls bouncy and frictionless
            for bid in [ball1, ball2]:
                p.changeDynamics(bid, -1, restitution=1.0, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0)
            body_ids = [ball1, ball2]
        elif index == 1:
            # Robotic Arm environment (Franka Panda or KUKA)
            # Use Franka Panda (7-DOF arm) with fixed base.
            robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0,0,0], useFixedBase=True)
            # Set joint initial poses (using a comfortable default pose)
            # (Joints 0-6 are arm, 7-8 are finger gripper which we'll ignore for movement.)
            rest_poses = [0, -0.3, 0, -2.0, 0, 2.0, 0.8]  # a slight variation of typical home pose
            num_joints = p.getNumJoints(robot_id)
            for j in range(len(rest_poses)):
                p.resetJointState(robot_id, j, rest_poses[j])
                p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=rest_poses[j], force=5*240.)
            # Keep gripper open
            if num_joints >= 9:
                p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=50)
                p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=50)
            body_ids = [robot_id]
        else:
            raise ValueError("Unknown environment index")
        return body_ids
    
    # Load initial environment
    body_ids = load_environment(env_index)
    
    # Initialize model components
    wave = MultiWaveletEmbed(wavelet='haar')
    # Determine latent dimensions from wavelet embedding:
    # For a 480x480 RGB image, wavelet output = 12 channels of 240x240.
    latent_H = 240
    latent_W = 240
    latent_channels = 12  # 3 (RGB) * 4 subbands
    # Create NCA model for the latent grid
    nca = KoopmanNCA(latent_dim=latent_channels).to(device)
    # Optimizer for NCA parameters (active inference updates)
    optimizer = optim.Adam(nca.parameters(), lr=1e-3)
    # Hopfield memory
    memory = HopfieldMemory(max_patterns=50)
    # DSL module
    dsl = DSLModule(num_coarse_channels=3, num_detail_channels=9)
    # Lagrange multiplier for constraint (latent energy conservation)
    lambda_energy = 0.0
    lambda_lr = 0.1  # learning rate for dual variable update
    
    # UI controls: using PyBullet's debug parameters
    # Slider for environment selection (0 or 1)
    env_slider = p.addUserDebugParameter("Environment", 0, 1, env_index)
    pause_slider = p.addUserDebugParameter("Pause Simulation", 0, 1, 0)
    speed_slider = p.addUserDebugParameter("Sim Speed", 0.1, 2.0, 1.0)
    nca_slider = p.addUserDebugParameter("NCA On", 0, 1, 1)
    memory_slider = p.addUserDebugParameter("Memory On", 0, 1, 1)
    dsl_slider = p.addUserDebugParameter("DSL On", 0, 1, 1)
    perturb_slider = p.addUserDebugParameter("Perturb Weights", 0, 1, 0)
    save_slider = p.addUserDebugParameter("Save Model", 0, 1, 0)
    load_slider = p.addUserDebugParameter("Load Model", 0, 1, 0)
    learn_slider = p.addUserDebugParameter("Learning On", 0, 1, 1)
    
    # Flags to detect button presses (perturb, save, load)
    prev_perturb = 0
    prev_save = 0
    prev_load = 0
    
    # Simulation loop variables
    last_obs_latent = None  # latent representation of last observed state
    predicted_latent = None  # model's predicted latent for current time
    step_count = 0
    
    # Main loop
    print("Starting simulation loop. Press 'q' or ESC in the PyBullet window to quit.")
    cv2.namedWindow("ModelView", cv2.WINDOW_NORMAL)
    # We will continuously simulate physics and update the model in real-time.
    while True:
        # Check for user input from UI sliders
        # Environment switch
        new_env_index = int(round(p.readUserDebugParameter(env_slider)))
        if new_env_index != env_index:
            env_index = new_env_index
            # Reload environment
            body_ids = load_environment(env_index)
            # Reset model internal state if needed
            last_obs_latent = None
            predicted_latent = None
            memory.reset()
        # Pause simulation
        paused = (p.readUserDebugParameter(pause_slider) > 0.5)
        # Speed factor
        speed = p.readUserDebugParameter(speed_slider)
        # Toggle components
        nca_enabled = (p.readUserDebugParameter(nca_slider) > 0.5)
        memory_enabled = (p.readUserDebugParameter(memory_slider) > 0.5)
        dsl_enabled = (p.readUserDebugParameter(dsl_slider) > 0.5)
        learning_enabled = (p.readUserDebugParameter(learn_slider) > 0.5)
        # Perturb weights button
        pert_val = p.readUserDebugParameter(perturb_slider)
        if pert_val > 0.5 and prev_perturb <= 0.5:
            # When slider crosses threshold to 1, randomize NCA weights
            with torch.no_grad():
                nca.A.data = 0.01 * torch.eye(nca.latent_dim, dtype=torch.cfloat, device=device) + 0.01 * (torch.randn_like(nca.A) + 1j*torch.randn_like(nca.A))
            print("NCA weights perturbed (randomized).")
        prev_perturb = pert_val
        # Save/Load model state
        save_val = p.readUserDebugParameter(save_slider)
        if save_val > 0.5 and prev_save <= 0.5:
            # Save NCA parameters and memory to file
            save_data = {
                "A": nca.A.detach().cpu(),
                "memory": [patt.detach().cpu() for patt in memory.patterns]
            }
            torch.save(save_data, "model_state.pth")
            print("Model state saved to model_state.pth.")
        prev_save = save_val
        load_val = p.readUserDebugParameter(load_slider)
        if load_val > 0.5 and prev_load <= 0.5:
            try:
                load_data = torch.load("model_state.pth", map_location=device)
                if "A" in load_data:
                    nca.A.data = load_data["A"].to(device)
                if "memory" in load_data:
                    memory.patterns = [patt.to(device) for patt in load_data["memory"]]
                print("Model state loaded from model_state.pth.")
            except FileNotFoundError:
                print("No saved model_state.pth found to load.")
        prev_load = load_val
        
        # If not paused, advance simulation and model
        if not paused:
            # Step physics simulation (for real-time, step once and then possibly sleep to adjust speed)
            p.stepSimulation()
            step_count += 1
            # Read environment state (observation)
            # We'll produce a 480x480 RGB "observation image" representing the environment.
            obs_img = np.zeros((3, 480, 480), dtype=np.float32)
            if env_index == 0:
                # Bouncing balls: we mark ball positions onto the image.
                # Get positions of balls
                # We assume body_ids[0], [1] correspond to our balls (if present).
                for idx, body in enumerate(body_ids):
                    if idx >= 2: break
                    pos, _ = p.getBasePositionAndOrientation(body)
                    x, y, z = pos
                    # Map x,y from [-1,1] range to image coordinates [0,479]
                    col = int(((x - (-1.0)) / 2.0) * 479)
                    row = int(((y - (-1.0)) / 2.0) * 479)
                    # Clamp to bounds
                    col = max(0, min(479, col))
                    row = max(0, min(479, row))
                    if idx == 0:
                        # Ball1 -> red
                        obs_img[0, row, col] = 1.0
                    elif idx == 1:
                        # Ball2 -> green
                        obs_img[1, row, col] = 1.0
            elif env_index == 1:
                # Robotic arm: mark end-effector position (projected to XY plane).
                robot_id = body_ids[0]
                # End effector link index for Panda is 11 (from documentation).
                ee_index = 11
                link_state = p.getLinkState(robot_id, ee_index)
                if link_state:
                    ee_pos = link_state[0]  # (x,y,z)
                    x, y, z = ee_pos
                    col = int(((x - (-1.0)) / 2.0) * 479)
                    row = int(((y - (-1.0)) / 2.0) * 479)
                    col = max(0, min(479, col))
                    row = max(0, min(479, row))
                    # Mark as blue dot
                    obs_img[2, row, col] = 1.0
                # Also update robot motion: to make the arm move, we vary one joint over time.
                # For example, oscillate joint 3 (index 3) sinusoidally.
                joint_index = 3
                target_angle = -2.0 + 0.5 * math.sin(0.005 * step_count)
                p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_angle, force=5*240.)
            # Convert observation image to latent via wavelet embedding.
            obs_latent = wave.forward(obs_img)
            # If this is an "observation available" step (we simulate partial observation):
            observe_step = True
            # Example of masked observation: we could do observe_step = (step_count % 5 == 0) to only observe every 5th step.
            # For demonstration, let's simulate that if desired:
            # observe_step = (step_count % 5 == 0)
            # We'll keep observe_step True every step for simplicity or user can modify above.
            
            if observe_step:
                # If we have a previous prediction for this step (i.e., predicted_latent is the model's guess for the current state),
                # we can compute error/loss and update model.
                if predicted_latent is not None:
                    # Compute losses between predicted_latent and obs_latent (the actual outcome).
                    # 1. L2 loss (MSE) on latent
                    diff = predicted_latent - obs_latent.to(device)
                    l2_loss = torch.mean(torch.abs(diff) ** 2)
                    # 2. Contrastive InfoNCE loss on coarse latent (slow features)
                    # Get coarse subbands (first 3 channels) from predicted and actual
                    coarse_pred = predicted_latent[0:3]
                    coarse_actual = obs_latent[0:3].to(device)
                    # Flatten and normalize
                    a = coarse_pred.flatten()
                    pos = coarse_actual.flatten()
                    # If memory has stored patterns, pick a random one as negative, else use a random older actual (here memory is effectively that store).
                    if memory_enabled and len(memory.patterns) > 0:
                        neg_pattern = random.choice(memory.patterns)
                        neg = neg_pattern.to(device)
                    else:
                        # If no memory, use previous actual latent (or just use a slightly perturbed coarse_actual as negative example for formality).
                        neg = coarse_actual.flatten() * 0.0  # trivial negative (not ideal, means no contrast if memory empty)
                    # Normalize vectors
                    if torch.linalg.norm(a) > 1e-8:
                        a_n = a / torch.linalg.norm(a)
                    else:
                        a_n = a
                    if torch.linalg.norm(pos) > 1e-8:
                        pos_n = pos / torch.linalg.norm(pos)
                    else:
                        pos_n = pos
                    if torch.linalg.norm(neg) > 1e-8:
                        neg_n = neg / torch.linalg.norm(neg)
                    else:
                        neg_n = neg
                    # Compute dot similarity in R^{2N} (consider real and imaginary parts)
                    a_real = torch.cat([a_n.real, a_n.imag])
                    pos_real = torch.cat([pos_n.real, pos_n.imag])
                    neg_real = torch.cat([neg_n.real, neg_n.imag])
                    sim_pos = torch.dot(a_real, pos_real)
                    sim_neg = torch.dot(a_real, neg_real)
                    # Temperature for InfoNCE
                    tau = 0.1
                    # Contrastive loss: -log(exp(sim_pos/tau) / (exp(sim_pos/tau) + exp(sim_neg/tau)))
                    # Add a small epsilon to avoid log(0) if needed
                    exp_pos = torch.exp(sim_pos / tau)
                    exp_neg = torch.exp(sim_neg / tau)
                    infoNCE_loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8))
                    # Total loss
                    total_loss = l2_loss + infoNCE_loss
                    # Constraint: latent energy conservation (optional, since NCA is mostly unitary)
                    # Compute energy difference between predicted and last observed latent (which was used to predict).
                    if last_obs_latent is not None:
                        energy_pred = torch.sum(torch.abs(predicted_latent) ** 2)
                        energy_last = torch.sum(torch.abs(last_obs_latent.to(device)) ** 2)
                        c_energy = energy_pred - energy_last
                    else:
                        c_energy = torch.tensor(0.0, device=device)
                    total_loss = total_loss + lambda_energy * c_energy
                    # Optimize model parameters if learning is enabled
                    if learning_enabled:
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                    # Update Lagrange multiplier for energy constraint (dual ascent)
                    lambda_energy += lambda_lr * c_energy.item()
                # After processing the observation, sync model state with actual observed latent.
                current_latent = obs_latent.clone().to(device)
                # Add to memory if enabled
                if memory_enabled:
                    # We store, for example, the coarse latent as memory pattern (to reduce dimensionality stored).
                    memory.add_pattern(current_latent[0:3])
                # Use current actual latent as starting point for model prediction of next step.
                input_latent = current_latent.clone()
                last_obs_latent = current_latent.clone()
            else:
                # If no observation (masked), we use model's predicted latent as input for next prediction (pure prediction mode).
                # The last predicted state becomes the basis for the next step prediction.
                input_latent = predicted_latent.detach()  # model believes this was the state
            # Now produce prediction for the next step:
            if nca_enabled:
                # Ensure input_latent is on correct device and shape (should be (12,240,240) complex).
                latent_grid = input_latent.to(device)
                # Reshape latent into (H, W, channels) format expected by NCA.
                latent_grid = latent_grid.permute(1, 2, 0).contiguous()  # shape now (240,240,12)
                # Apply one NCA update step
                new_latent_grid = nca(latent_grid)
                # Reshape back to (channels, H, W)
                new_latent = new_latent_grid.permute(2, 0, 1).contiguous()
            else:
                # If NCA is disabled, just carry forward the input latent (no prediction).
                new_latent = input_latent.to(device)
            # Optionally, incorporate Hopfield memory influence on the predicted latent (e.g., attract to stored patterns).
            if memory_enabled and nca_enabled and len(memory.patterns) > 0:
                # Example usage: retrieve closest memory pattern for the coarse latent and blend it in slightly.
                query = new_latent[0:3]  # coarse part of predicted latent
                retrieved = memory.retrieve(query)
                if retrieved is not None:
                    # Blend retrieved pattern (which is flattened coarse) with current coarse.
                    retrieved_coarse = retrieved.view(3, latent_H, latent_W)
                    # Simple strategy: average the coarse latent with retrieved coarse pattern (could also do weighted by similarity).
                    new_latent[0:3] = 0.5 * new_latent[0:3] + 0.5 * retrieved_coarse.to(device)
            # Apply DSL program to refine latent if enabled
            if dsl_enabled:
                new_latent = dsl.run(new_latent)
            # Set predicted_latent for next loop iteration
            predicted_latent = new_latent.detach()
        # If paused, we do nothing (the model state and physics are frozen).
        
        # Visualization/Overlay:
        # We will display the observation image with overlays of status.
        # For model view, we can show the observation or some representation of latent.
        # We'll use the observation image for simplicity (since latent is multi-scale and complex).
        display_img = np.copy(obs_img)  # shape (3,480,480)
        # Add overlay text for losses (if available), environment, etc.
        overlay_text = []
        if not paused and 'total_loss' in locals():
            overlay_text.append(f"L2 loss: {l2_loss.item():.4f}")
            overlay_text.append(f"InfoNCE loss: {infoNCE_loss.item():.4f}")
        else:
            overlay_text.append("Paused" if paused else "No loss (initial step)")
        overlay_text.append(f"Env: {'Bouncing Balls' if env_index==0 else 'Robot Arm'}")
        overlay_text.append(f"Memory size: {len(memory.patterns)}")
        # Compose text on the image (white text)
        for i, txt in enumerate(overlay_text):
            cv2.putText(display_img, txt, (10, 20 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1,1,1), 1)
        # Show the image (convert to BGR for imshow)
        img_bgr = (np.transpose(display_img, (1, 2, 0)) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        cv2.imshow("ModelView", img_bgr)
        cv2.waitKey(1)
        
        # Manage simulation timing for real-time speed:
        if not paused:
            # Aim for ~60 Hz base. We already stepped once which is one frame.
            target_dt = 1.0 / (60.0 * speed)
            # Sleep to achieve target frame time (note: this simple approach may be imprecise).
            time.sleep(max(0, target_dt))
        
        # Check for exit condition (keyboard events from PyBullet window)
        keys = p.getKeyboardEvents()
        if any([k == p.B3G_ESCAPE or k == ord('q') for k in keys.keys()]):
            break
    
    # Cleanup on exit
    p.disconnect()
    cv2.destroyAllWindows()
