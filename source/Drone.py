class Drone():
    """!
    This is a general parent class for the drones.
    For any particular model the user should create a subclass and specify the functions below.
    """

    def __init__(self, fom):
        """! Initialization for the drone class
        In this call we specify the main setup parameters that the drone class has to have.
        The user needs to specify their own __init__ call, and call super().__init__ for the setup here.

        @param fom  Full-order-model (FOM) object. The drone needs to measure states computed by the FOM
        """
        # TODO: are there any other setup parameters?
        self.fom = fom

    # TODO: specify and describe other functions the general drone class has to have