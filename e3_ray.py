import random
import sys
from pathlib import Path
from typing import Final

import gymnasium as gym
import ray
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios

# Số lượng agents trong kịch bản
N_AGENTS = 4
AGENT_IDS: Final[list] = ["Agent %i" % i for i in range(N_AGENTS)]


class RandomLanerAgent(Agent):
    """Agent thực hiện hành động ngẫu nhiên."""
    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, obs, **kwargs):
        return self._action_space.sample()


@ray.remote
def run_scenario(scenario, num_episodes, max_episode_steps, agent_ids):
    """Hàm chạy mô phỏng trên một máy với Ray."""
    # Khởi tạo AgentInterface
    agent_interfaces = {
        agent_id: AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        )
        for agent_id in agent_ids
    }

    # Tạo môi trường SMARTS với Envision
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=[scenario],
        agent_interfaces=agent_interfaces,
        headless=False,  # Hiển thị giao diện Envision
    )

    # Khởi tạo các agent
    agents = {agent_id: RandomLanerAgent(env.action_space[agent_id]) for agent_id in agent_ids}

    # Biến lưu tổng score cho từng agent qua các episodes
    total_scores = {agent_id: 0 for agent_id in agent_ids}

    for episode in episodes(n=num_episodes):
        observations, _ = env.reset()
        terminateds = {"__all__": False}
        episode_scores = {agent_id: 0 for agent_id in agent_ids}  # Score của mỗi agent trong episode

        while not terminateds["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(observations[agent_id])
                for agent_id in observations
            }
            observations, rewards, terminateds, truncateds, infos = env.step(actions)

            # Cộng dồn score của từng agent trong episode
            for agent_id, reward in rewards.items():
                episode_scores[agent_id] += reward

        # Cập nhật tổng score của từng agent sau episode
        for agent_id in agent_ids:
            total_scores[agent_id] += episode_scores[agent_id]

        # Tính và in score trung bình của từng agent sau episode
        avg_scores = {agent_id: episode_scores[agent_id] / max_episode_steps for agent_id in agent_ids}
        print(f"Episode finished! Average scores: {avg_scores}")

    env.close()

    return f"Completed scenario: {scenario}"



def main():
    # Đường dẫn tới repo SMARTS
    SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
    sys.path.insert(0, str(SMARTS_REPO_PATH))

    from examples.tools.argument_parser import minimal_argument_parser

    # Phân tích các đối số từ command line
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
        ]

    # Xây dựng kịch bản nếu chưa có
    build_scenarios(scenarios=args.scenarios)

    # Khởi tạo Ray
    ray.init()

    # Cấu hình các kịch bản
    scenarios = [
        {"scenario": str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop")},
        {"scenario": str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "custom")},
    ]

    # Chạy mô phỏng trên mỗi máy
    results = []
    for scenario in scenarios:
        result = run_scenario.remote(
            scenario=scenario["scenario"],
            num_episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            agent_ids=AGENT_IDS,
        )
        results.append(result)

    # Chờ tất cả các kịch bản hoàn thành
    ray.get(results)

    # Dừng Ray sau khi hoàn tất
    ray.shutdown()


if __name__ == "__main__":
    main()
