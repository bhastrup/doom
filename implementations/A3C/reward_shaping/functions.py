last_total_health = 100
last_total_ammo2 = 26  # total is 26 for defend the center scenario

while not env.is_episode_finished():
  ammo2_delta = env.get_game_variable(GameVariable.AMMO2) - last_total_ammo2
  last_total_ammo2 = env.get_game_variable(GameVariable.AMMO2)

  health_delta = env.get_game_variable(GameVariable.HEALTH) - last_total_health
  last_total_health = env.get_game_variable(GameVariable.HEALTH)

def health_reward_function(health_delta):
    health, reward = env.get_game_variable(GameVariable.HEALTH), 0
    if health_delta == 0:
        return 0
    elif health_delta < 0:
        reward = -0.5
    return reward
    
def ammo2_reward_function(ammo2_delta):
    if ammo2_delta == 0:
        return 0
    elif ammo2_delta > 0:
        return -0.05
    else:
        return -0.05
