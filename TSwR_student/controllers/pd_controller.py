import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot):
        ### TODO: Please implement me

        vectors = [q, q_dot, q_d, q_d_dot]
        desired_length = 4
        for i, vector in enumerate(vectors):
            padding_length = desired_length - len(vector)
            vectors[i] = np.pad(vector, (0, padding_length), mode='constant')

        q, q_dot, q_d, q_d_dot = vectors
        
        e = q_d - q
        e_dot = q_d_dot - q_dot
        u = self.kp * e + self.kd * e_dot
        return u
