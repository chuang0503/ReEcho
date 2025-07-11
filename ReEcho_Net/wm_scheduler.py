import math

class TauTeacherScheduler:
    def __init__(self, tau_init=0.5, tau_min=0.1,
                 t_init=1, total_steps=100_000, k_tau=3.0):
        self.tau_init, self.tau_min = tau_init, tau_min
        self.t_init = t_init
        self.total  = total_steps
        self.k_tau  = k_tau
        self.step   = 0

    def update(self):
        self.step += 1
        p   = self.step / self.total
        tau = max(self.tau_min, self.tau_init * math.exp(-self.k_tau * p))
        t_force = max(0., self.t_init * (1 - p))
        return tau, t_force
    

def test_wm_scheduler():
    scheduler = TauTeacherScheduler(total_steps=1000)
    for i in range(1000):
        tau, t_force = scheduler.update()
        print(f"Step {i}: tau={tau}, t_force={t_force}")

if __name__ == "__main__":
    test_wm_scheduler()