
class TemperatureScheduler:
    def __init__(self, initial_temp, step_size, gamma):
        self.step_size = step_size
        self.temperature = initial_temp
        self.current_step = 0
        self.gamma = gamma
    
    def step(self):
        self.current_step += 1
        if (self.current_step != 0) and (self.current_step % self.step_size == 0):
            self.temperature *= self.gamma

    def get_temperature(self):
        return self.temperature