# sympy (symbolic solutions)
def example_1():
    from sympy import symbols, solve

    x, y = symbols('x y')
    equation = x**2 - 4
    solutions = solve(equation, x)
    print(solutions) # Output: [-2, 2]


# scipt.optimize.fsolve (not ODE)
def example_2():
    from scipy.optimize import fsolve
    import math

    def func(x):
        return [math.cos(x[0]) - x[1], x[0] - x[1]**2]

    initial_guess = [0, 0]
    solution = fsolve(func, initial_guess)
    print(solution)


# scipt.optimize.fsolve (not ODE)
def example_3():
    from scipy.optimize import fsolve
    import numpy as np

    def equations(p):
        x, y = p
        return (x**2 + y**2 - 4, x**2 - y - 2)

    x0 = np.array([1, 1])  # Initial guess
    solution = fsolve(equations, x0)
    print(f"Solution: x = {solution[0]}, y = {solution[1]}")


# scipt.optimize.fsolve (not ODE)
def example_4():
    from scipy.optimize import fsolve
    import numpy as np

    def equations(p):
        t1, t2, t3, t4 = p
        return (
            x**2 + y**2 - 4,
            x**2 - y - 2,
        )

    x0 = np.array([1, 1])  # Initial guess
    solution = fsolve(equations, x0)
    print(f"Solution: x = {solution[0]}, y = {solution[1]}")


# scipy.integrate.solve_ivp (ODE system!)
def example_5():
    import numpy as np
    from scipy.integrate import solve_ivp

    # Define the ODE system (e.g., Lotka-Volterra equations)
    def lotkavolterra(t, z, a, b, c, d):
        x, y = z
        return [a * x - b * x * y, -c * y + d * x * y]

    # Initial conditions and time span
    y0 = [10, 5]  # Initial populations of prey (x) and predator (y)
    t_span = (0, 15)

    # Parameters for the Lotka-Volterra system
    args = (1.5, 1, 3, 1)

    # Solve the IVP
    sol = solve_ivp(lotkavolterra, t_span, y0, args=args, dense_output=True)

    # Access the solution
    print("Times:", sol.t)
    print("Solution at computed times:\n", sol.y)

    # Evaluate the solution at specific times (using dense_output)
    t_eval = np.linspace(0, 15, 100)
    y_interp = sol.sol(t_eval)


def example_6():
    # scipy.integrate.solve_ivp (ODE system!)
    # Thermal simulation - 2 nodes
    import numpy as np
    from scipy.integrate import solve_ivp

    # Define constants
    SF = 5.67e-8 # W / (m2 * K4)
    I_SUN = 1400 # [W/m2]
    m1 = 10 # kg
    c1 = 444 # (Fe) Csp = 0.444 J/(g*°C) = 0.444 kg*m2/s2 * 1000g/kg / (g*°C) * °K/°C = 444 m2/s2 / °K
    emm_1 = 0.8
    area_rad_1 = 1 # 1 m2
    abs_1 = 0.2
    area_in_1_eff = 0.707  # 1 m2 * cos(45)
    cond_12 = 0.1 # W / (m2 * K)
    area_cond_12 = 0.1 # m2
    m2 = 10 # kg
    c2 = 444 # (Fe) Csp = 0.444 J/(g*°C) = 0.444 kg*m2/s2 * 1000g/kg / (g*°C) * °K/°C = 444 m2/s2 / °K
    emm_2 = 0.7
    area_rad_2 = 1 # 2 m2
    abs_2 = 0.3
    area_in_2_eff = 0.707  # 1 m2 * cos(45)
    area_cond_21 = area_cond_12
    cond_21 = cond_12

    # Define the ODE system (Thermal equations)
    def eq_system(t, y):
        T1, T2 = y
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

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(t_eval, y_interp[0], label='T1')
    plt.plot(t_eval, y_interp[1], label='T2')
    plt.xlabel('t')
    # plt.legend(['T1', 'T2'], shadow=True)
    plt.legend(shadow=True)
    plt.title('Thermal System')
    plt.show()
    fig.savefig('lala.png')



def example_7():
    # THERMAL EMISSIVITY; SOLAR ABSOR..
    # Clarifica mis dudas!!
    # https://www.electronics-cooling.com/2025/03/radiation-basics-making-sense-of-emissivity-absorptivity/

    # scipy.integrate.solve_ivp (ODE system!)
    # Thermal simulation - 2 nodes
    import numpy as np
    from scipy.integrate import solve_ivp

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

    import math
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

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(t_eval, y_interp[0], label='T1')
    plt.plot(t_eval, y_interp[1], label='T2')
    plt.xlabel('t')
    # plt.legend(['T1', 'T2'], shadow=True)
    plt.legend(shadow=True)
    plt.title('Thermal System')
    plt.show()
    fig.savefig('lala.png')

