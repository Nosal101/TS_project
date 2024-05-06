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
        differences = [np.linalg.norm(x - model.state) for model in self.models]
        self.i = np.argmin(differences)
        print(self.models[self.i].m3)
        

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        #v = q_r_ddot # TODO: add feedback
        Kd = 1
        Kp = 1
        v = q_r_ddot + Kd*(q_r_dot - q_dot) + Kp*(q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
