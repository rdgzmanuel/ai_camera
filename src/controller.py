from typing import Optional
from src.utils import enforce_aspect_ratio


class PIDController:
    """
    A simple PID controller to smoothly transition a value towards a target.

    Attributes:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        max_output (Optional[float]): Optional limit on the absolute value of output.
    """

    def __init__(self, kp: float, ki: float, kd: float, max_output: Optional[float] = None) -> None:
        """
        Initializes the PID controller.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            max_output: Optional maximum output value (for clamping).
        """
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.max_output: Optional[float] = max_output

        self.integral: float = 0.0
        self.prev_error: float = 0.0


    def reset(self) -> None:
        """
        Resets the integral and derivative state.
        """
        self.integral = 0.0
        self.prev_error = 0.0
        return None


    def update(self, target: float, current: float) -> float:
        """
        Computes the PID output for the given target and current values.

        Args:
            target (float): The desired value.
            current (float): The current value.

        Returns:
            The output adjustment based on PID control.
        """
        error: float = target - current
        self.integral += error
        derivative: float = error - self.prev_error

        output: float = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.max_output is not None:
            output = max(-self.max_output, min(self.max_output, output))

        self.prev_error = error
        return output


class BoxController:
    """
    Controls smooth transitions of a bounding box using individual PID controllers
    for center (cx, cy) and dimensions (w, h).

    Attributes:
        cx_pid (PIDController): PID controller for horizontal center.
        cy_pid (PIDController): PID controller for vertical center.
        w_pid (PIDController): PID controller for width.
        h_pid (PIDController): PID controller for height.
        current_box (Optional[Tuple[float, float, float, float]]): Current smoothed box state.
    """

    def __init__(
        self,
        target_ratio: float,
        kp: float = 0.01,
        ki: float = 0.0075,
        kd: float = 0.03,
        max_speed: float = 1.5
    ) -> None:
        """
        Initializes the BoxController with PID controllers for each component.

        Args:
            kp: Proportional gain for all components (how strongly it corrects errors).
            ki: Integral gain for all components (how much it "accumulates" error).
            kd: Derivative gain for all components (how strongly it resists overshoot).
            max_speed: Maximum allowed movement per update step.
        """
        self.cx_pid: PIDController = PIDController(kp, ki, kd, max_output=max_speed)
        self.cy_pid: PIDController = PIDController(kp, ki, kd, max_output=max_speed)
        self.w_pid: PIDController = PIDController(kp, ki, kd, max_output=max_speed)
        self.h_pid: PIDController = PIDController(kp, ki, kd, max_output=max_speed)

        self.target_ratio: float = target_ratio

        self.current_box: Optional[tuple[float, float, float, float]] = None


    def reset(self) -> None:
        """
        Resets all internal PID controllers and current box state.
        """
        self.cx_pid.reset()
        self.cy_pid.reset()
        self.w_pid.reset()
        self.h_pid.reset()
        self.current_box = None
        return None


    def update(self, target_box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """
        Updates the current box values towards the target using PID control.

        Args:
            target_box: Target box as (cx, cy, w, h).

        Returns:
            Updated box values as (cx, cy, w, h).
        """
        if self.current_box is None:
            self.current_box = target_box
            return target_box

        cx, cy, w, h = self.current_box
        target_cx, target_cy, target_w, target_h = target_box

        cx += self.cx_pid.update(target_cx, cx)
        cy += self.cy_pid.update(target_cy, cy)
        w += self.w_pid.update(target_w, w)
        h += self.h_pid.update(target_h, h)

        cx, cy, w, h = enforce_aspect_ratio(cx, cy, w, h, self.target_ratio)

        self.current_box = (cx, cy, w, h)
        return self.current_box


