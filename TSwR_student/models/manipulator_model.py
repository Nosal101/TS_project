import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 1.0
        self.r3 = 0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        self.d1 = self.l1/2
        self.d2 = self.l2/2

        self.update_dependent_values()
        #self.alpha = self.m1*(self.d1**2) + self.I_1 + self.m2*(self.l1**2 + self.d2**2) + self.I_2 
        #self.betha = self.m2*self.l1*self.d2
        #self.gama = self.m2*(self.d2**2) + self.I_2

        # self.alpha = self.m1*(self.d1**2) + self.I_1 + self.m2*(self.l1**2 + self.d2**2) + self.I_2 + self.m3*(self.l1**2 + self.l2**2) + self.I_3 
        # self.betha = self.m2*self.l1*self.d2 + self.m3*self.l1*self.l2
        # self.gama = self.m2*(self.d2**2) + self.I_2 + self.m3*(self.l2**2) + self.I_3


    def update_dependent_values(self):
        self.alpha = self.m1 * (self.d1 ** 2) + self.I_1 + self.m2 * (self.l1 ** 2 + self.d2 ** 2) + self.I_2 + self.m3 * (self.l1 ** 2 + self.l2 ** 2) + self.I_3
        self.betha = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        self.gama = self.m2 * (self.d2 ** 2) + self.I_2 + self.m3 * (self.l2 ** 2) + self.I_3

    def set_m3(self, m3):
        self.m3 = m3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.update_dependent_values()

    def set_r3(self, r3):
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.update_dependent_values()
        
    def M(self, x):

        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        self.macierz = np.array([[self.alpha + 2*self.betha*np.cos(q2), self.gama + self.betha*np.cos(q2)],
                                 [self.gama + self.betha*np.cos(q2), self.gama]]) 
        return self.macierz

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        
        self.macierz = np.array([[-1*self.betha*np.sin(q2)*q2_dot, -1*self.betha*np.sin(q2)*(q1_dot+q2_dot)],
                                 [self.betha*np.sin(q2)*q1_dot, 0]])
        return self.macierz
