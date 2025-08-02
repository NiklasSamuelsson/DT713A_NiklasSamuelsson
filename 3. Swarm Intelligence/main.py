from agent_creator import ProtectiveAgent1Creator
from agent_creator import ProtectiveAgent2Creator
from agent_creator import HidingAgent1Creator
from agent_creator import HidingAgent2Creator
from environment import Environment


if __name__ == "__main__":
    #agent_creator = ProtectiveAgent1Creator()
    #agent_creator = ProtectiveAgent2Creator()
    agent_creator = HidingAgent1Creator(20)
    #agent_creator = HidingAgent2Creator()
    env = Environment(1000, 10.0, (100, 100), agent_creator)
    env.reset("random")
    env.run((-2000, 2100), (-2000, 2100))
    #env.run()
