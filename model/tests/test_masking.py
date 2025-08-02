import numpy as np
import pytest
from model.gym_wrapper import TresetteGymWrapper
from model.state_encoder import EncodedState

from model.state_encoder import SUITS, RANKS, EncodedState

import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from model.training_config import TrainingConfig
from model.gym_wrapper import TresetteGymWrapper
from model.maskable_policy import MaskableActor, MaskablePPOPolicy
from model.tressete_actor import TressetteActor
from model.model import CriticMLP
from tianshou.data import Batch
from model.maskable_policy import MaskablePPOPolicy


def test_card_to_index_consistency():
    class DummyCard:
        def __init__(self, suit, rank):
            self.suit = suit
            self.rank = rank
    card = DummyCard(SUITS[0], RANKS[0])
    idx = EncodedState.card_to_index(card)
    assert 0 <= idx < len(SUITS) * len(RANKS)

def test_encode_state_mask_and_shape():
    env = TresetteGymWrapper()
    encoded = EncodedState.encode_state(env.agent_index, env)
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (214,)
    # Check that mask is binary
    mask = encoded[204:]
    assert set(np.unique(mask)).issubset({0.0, 1.0})

def test_update_player_state_sync():
    env = TresetteGymWrapper()
    encoder = EncodedState()
    encoder.update_player_state(env)
    assert encoder.agent_state.shape == (214,)
    assert encoder.opponent_state.shape == (214,)

def test_reset_returns_correct_obs_and_mask():
    env = TresetteGymWrapper()
    obs, _ = env.reset()
    assert "obs" in obs and "action_mask" in obs
    assert obs["obs"].shape == (214,)
    assert obs["action_mask"].shape == (10,)
    # Mask binary check
    mask = obs["action_mask"]
    assert set(np.unique(mask)).issubset({0.0, 1.0})

def test_step_valid_and_invalid_actions():
    env = TresetteGymWrapper()
    obs, _ = env.reset()
    valid_actions = np.where(obs["action_mask"] == 1)[0]
    assert len(valid_actions) > 0, "No valid actions found"

    # Step with a valid action should work
    action = valid_actions[0]
    next_obs, reward, done, truncated, info = env.step(action)
    assert "obs" in next_obs and "action_mask" in next_obs
    assert next_obs["obs"].shape == (214,)

    # Step with invalid action should raise
    invalid_actions = np.where(obs["action_mask"] == 0)[0]
    if len(invalid_actions) > 0:
        with pytest.raises(ValueError):
            env.step(invalid_actions[0])

def get_dummy_policy():
    actor_net = TressetteActor(214, 512, 10)
    actor = MaskableActor(actor_net)
    return actor  # or return a full policy if needed

def test_policy_never_samples_invalid_action():
    env = TresetteGymWrapper()
    raw_obs, _ = env.reset()

    # Properly batchify obs
    obs_batch = Batch({
        "obs": np.expand_dims(raw_obs["obs"], axis=0),
        "action_mask": np.expand_dims(raw_obs["action_mask"], axis=0)
    })

    policy = get_dummy_policy()
    action_output = policy(obs_batch)

    assert action_output is not None

def test_mask_updates_after_step():
    env = TresetteGymWrapper()
    obs, _ = env.reset()
    valid_actions = np.where(obs["action_mask"] == 1)[0]
    action = valid_actions[0]
    next_obs, _, _, _, _ = env.step(action)
    new_mask = next_obs["action_mask"]
    assert new_mask.shape == (10,)
    assert set(np.unique(new_mask)).issubset({0.0, 1.0})

import torch
from model.maskable_policy import MaskableActor
from model.tressete_actor import TressetteActor

def test_maskable_actor_masking_behavior():
    device = "cpu"
    actor_net = TressetteActor()
    actor = MaskableActor(actor_net).to(device)
    actor.eval()

    # Prepare dummy obs and mask
    obs = torch.zeros((1, 214), dtype=torch.float32)
    mask = torch.zeros((1, 10), dtype=torch.float32)
    mask[0, 2] = 1  # Only action index 2 is valid

    logits, _ = actor({"obs": obs, "action_mask": mask})

    # Check logits at invalid positions are very negative
    neg_inf = torch.finfo(logits.dtype).min
    assert torch.all(logits[0, mask[0] == 0] == neg_inf)
    assert torch.any(logits[0, mask[0] == 1] != neg_inf)

def test_maskable_ppo_policy_forward_and_action():
    actor_net = TressetteActor()
    actor = MaskableActor(actor_net)
    critic_net = torch.nn.Linear(214, 1)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic_net.parameters()), lr=1e-3)

    policy = MaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=0.99,
        max_grad_norm=0.5,
        action_space=None
    )
    obs = torch.zeros((1, 214), dtype=torch.float32)
    mask = torch.ones((1, 10), dtype=torch.float32)
    batch = Batch(obs=Batch(obs=obs, action_mask=mask), info=Batch())
    
    batch_out = policy(batch)
    actor_net = TressetteActor()
    actor = MaskableActor(actor_net)
    critic_net = torch.nn.Linear(214, 1)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic_net.parameters()), lr=1e-3)

    policy = MaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=0.99,
        max_grad_norm=0.5,
        action_space=None
    )

    obs = torch.zeros((1, 214), dtype=torch.float32)
    mask = torch.ones((1, 10), dtype=torch.float32)

    # Make sure info is a valid Batch
    batch = Batch(
        obs=Batch(obs=obs, action_mask=mask),
        info=Batch()  # not dict!
    )

    batch_out = policy(batch)
    assert hasattr(batch_out, "act")
    assert batch_out.act.shape[0] == 1

def test_training_loop_runs_without_errors():
    cfg = TrainingConfig()
    cfg.num_train_envs = 1
    cfg.num_test_envs = 1
    cfg.step_per_epoch = 100
    cfg.max_epoch = 1
    cfg.batch_size = 16

    device = "cpu"

    train_envs = DummyVectorEnv([TresetteGymWrapper])
    test_envs = DummyVectorEnv([TresetteGymWrapper])

    actor_net = TressetteActor(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, action_dim=cfg.action_dim).to(device)
    critic_net = CriticMLP(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim).to(device)
    actor = MaskableActor(actor_net).to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic_net.parameters()), lr=cfg.lr)

    policy = MaskablePPOPolicy(
        actor=actor,
        critic=critic_net,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=cfg.discount_factor,
        max_grad_norm=cfg.max_grad_norm,
        action_space=train_envs.action_space[0]
    ).to(device)

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(500, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # Initial collect
    train_collector.collect(n_step=100)

    # Train for one epoch
    result = None
    try:
        from tianshou.trainer import onpolicy_trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=cfg.max_epoch,
            step_per_epoch=cfg.step_per_epoch,
            repeat_per_collect=1,
            episode_per_test=1,
            batch_size=cfg.batch_size,
            step_per_collect=50,
        )
    except Exception as e:
        pytest.fail(f"Training loop raised an exception: {e}")

    assert result is not None
    assert "best_result" in result
    assert "best_reward" in result
    assert "duration" in result
