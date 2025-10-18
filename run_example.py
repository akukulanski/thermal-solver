import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use('qtagg')


def main():
    # THERMAL EMISSIVITY; SOLAR ABSOR..
    # Clarifica mis dudas!!
    # https://www.electronics-cooling.com/2025/03/radiation-basics-making-sense-of-emissivity-absorptivity/

    # scipy.integrate.solve_ivp (ODE system!)
    # Thermal simulation - 2 nodes

    # Define constants
    SF = 5.67e-8 # W / (m2 * K4)
    I_SUN = 1400 # [W/m2]
    m1 = 10 # kg
    c1 = 444 # (Fe) Csp = 0.444 J/(g*°C) = 0.444 kg*m2/s2 * 1000g/kg / (g*°C) * °K/°C = 444 m2/s2 / °K
    emm_1 = 0.8
    area_rad_1 = 1 # 1 m2
    abs_1 = 0.2
    # area_in_1_eff = 0.707  # 1 m2 * cos(45)
    area_in_1_exposed = 1  # 1 m2
    cond_12 = 0.1 # W / (m2 * K)
    area_cond_12 = 0.1 # m2
    m2 = 10 # kg
    c2 = 444 # (Fe) Csp = 0.444 J/(g*°C) = 0.444 kg*m2/s2 * 1000g/kg / (g*°C) * °K/°C = 444 m2/s2 / °K
    emm_2 = 0.7
    area_rad_2 = 1 # 2 m2
    abs_2 = 0.3
    # area_in_2_eff = 0.707  # 1 m2 * cos(45)
    area_in_2_exposed = 1 # 1 m2
    area_cond_21 = area_cond_12
    cond_21 = cond_12

    def sun_angle_area_1(t: float) -> float:
        period = 3600 # 1 h
        freq = 1 / period
        return 2 * math.pi * freq * t

    def get_area_in_1_eff(t: float) -> float:
        factor = math.cos(sun_angle_area_1(t))
        factor = max(factor, 0)  # never below 0
        return area_in_1_exposed * factor

    def sun_angle_area_2(t: float) -> float:
        return math.pi / 3  # constant to 60 deg

    def get_area_in_2_eff(t: float) -> float:
        factor = math.cos(sun_angle_area_2(t))
        factor = max(factor, 0)  # never below 0
        return area_in_2_exposed * factor

    # Define the ODE system (Thermal equations)
    def eq_system(t, y):
        T1, T2 = y
        area_in_1_eff = get_area_in_1_eff(t)
        area_in_2_eff = get_area_in_2_eff(t)
        return [
            # - m1 * c1 * dT1/dt = SF * emm_1 * area_rad_1 * T1**4 - abs_1 * area_in_1_eff * I_SUN - cond_12 * area_cond_12 * (T1 - T2)
            - (1 / m1 / c1) * (SF * emm_1 * area_rad_1 * T1**4 - abs_1 * area_in_1_eff * I_SUN - cond_12 * area_cond_12 * (T1 - T2)),
            # - m2 * c2 * dT2/dt = SF * emm_2 * area_rad_2 * T2**4 - abs_2 * area_in_2_eff * I_SUN - cond_21 * area_cond_21 * (T2 - T1)
            - (1 / m2 / c2) * (SF * emm_2 * area_rad_2 * T2**4 - abs_2 * area_in_2_eff * I_SUN - cond_21 * area_cond_21 * (T2 - T1)),
        ]

    # Initial conditions and time span
    y0 = [300, 300]
    t_span = (0, 3600 * 4)

    # Parameters for the Lotka-Volterra system
    args = ()

    # Solve the IVP
    sol = solve_ivp(fun=eq_system, t_span=t_span, y0=y0, args=args, dense_output=True)

    # Access the solution
    print("Times:", sol.t)
    print("Solution at computed times:\n", sol.y)

    # Evaluate the solution at specific times (using dense_output)
    t_eval = np.linspace(*t_span, 100)
    y_interp = sol.sol(t_eval)

    fig = plt.figure()
    plt.plot(t_eval, y_interp[0], label='T1')
    plt.plot(t_eval, y_interp[1], label='T2')
    plt.xlabel('t')
    # plt.legend(['T1', 'T2'], shadow=True)
    plt.legend(shadow=True)
    plt.title('Thermal System')
    plt.show()
    fig.savefig('lala.png')


if __name__ == '__main__':
    main()