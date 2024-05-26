import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        #self.models = [None, None, None]
        self.Tp = Tp
        self.i = 0
        self.Kp = 2
        self.Kd = 2
        self.u_prev = [0, 0]
        self.x_prev = [0, 0, 0, 0]
        self.first = True

        self.models = [
            self.create_manipulator_model(0.1, 0.05),
            self.create_manipulator_model(0.01, 0.01),
            self.create_manipulator_model(1.0, 0.3)]

    def create_manipulator_model(self, m3, r3):
        manipulator_model = ManiuplatorModel(self.Tp)
        manipulator_model.set_m3(m3)
        manipulator_model.set_r3(r3)
        return manipulator_model
    

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        high_error = 999999999
        for i in range(0,3):
            q_dot_dot = np.linalg.solve(self.models[i].M(x), self.u_prev - self.models[i].C(x) @ x[2:])
            q_dot = self.x_prev[2:] + q_dot_dot * self.Tp
            q = x[:2] + q_dot * self.Tp

            x_estimate = np.concatenate([q , q_dot])

            error = abs(x_estimate[0] - x[0]) + abs(x_estimate[1] - x[1]) + abs(x_estimate[2] - x[2]) + abs(x_estimate[3] - x[3])
        
            if error < high_error:
                high_error = error
                self.i = i

        print("Model chosen: ", self.i, "Error: ", high_error)
            
        

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        if self.first:
            self.first = False
            self.x_prev = x
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        #v = q_r_ddot # TODO: add feedback
        v = q_r_ddot + self.Kd*(q_r_dot - q_dot) + self.Kp*(q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot
        self.u_prev = u
        self.x_prev = x
        return u
