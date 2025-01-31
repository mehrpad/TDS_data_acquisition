





class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        """
        Initializes the PID controller.

        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param setpoint: Desired temperature
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self.previous_error = 0
        self.integral = 0
        self.output = 0

    def update_setpoint(self, setpoint):
        """
        Update the setpoint of the controller.

        :param setpoint: The new desired temperature
        """
        self.setpoint = setpoint

    def compute(self, current_temperature, setpoint=None):
        """
        Compute the PID control output based on the current temperature.

        :param current_temperature: The current temperature of the system
        :param setpoint: The desired temperature (optional)
        :return: The control output to adjust the temperature
        """
        # Calculate the error
        if setpoint is not None:
            self.setpoint = setpoint
        error = self.setpoint - current_temperature

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error)

        # Calculate the output
        print(f"Proportional: {proportional}, Integral: {integral}, Derivative: {derivative}")
        self.output = proportional + integral + derivative

        # Save the error for the next iteration
        self.previous_error = error

        return self.output