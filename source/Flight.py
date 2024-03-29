import numpy as np

class Flight():
    
    dpdc = None # d position d control
    dmdc = None # d measurement d control
    measurement = None # measurement along this flight path
    
    def __init__(self, navigation, alpha):
        
        self.navigation = navigation
        self.alpha = alpha
        self.flightpath, self.grid_t = navigation.get_trajectory(alpha)
        
    def get_position(self, t: float | np.ndarray):
        """! Get the position of the drone given the time and flying parameters

        @param t  The time to evaluate the position of the drone
        @param alpha  The parameters of the flight path
        @return  spatial position of the drone
        """
        print("WARNING: Navigation.get_position was called. Should be replaced with flight.get_position")
        
        pos, __ = self.naviation.get_trajectory(alpha=self.alpha, grid_t=t * np.ones((1,)))
        return pos[0, :]
    
    def d_position_d_control(self):
        if self.dpdc is not None:
            return self.dpdc
        
        self.dpdc = self.navigation.d_position_d_control(flight=self)
        return self.dpdc