# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backwards-compatibility shim: CrowdSim.ppo_policy → CrowdSim.ppo.ppo_policy.

Checkpoints saved before the module was reorganised into CrowdSim/ppo/
contain pickle references to ``CrowdSim.ppo_policy.*``.  Re-exporting
everything from the new location lets ``torch.load(..., weights_only=False)``
find the classes without having to re-save the checkpoint.
"""

from CrowdSim.ppo.ppo_policy import (  # noqa: F401
    DepthEncoder,
    RobotActorCritic,
    RobotPPOConfig,
    RobotPPOTrainer,
    RobotRolloutBuffer,
    bounded_robot_action,
    bounded_robot_action_log_prob,
    expand_or_validate,
    make_conv_encoder,
    make_mlp,
    network_hidden_dims,
    normalize_dims,
    optional_hidden_dims,
    robot_network_kwargs,
)
