
# Taylor's formula
f(x_0) = f(x) + f'(x) * (x-x_0) + o(x_x_0)

# simplified
x_0  = x - f(x) / f'(x)

# Problem
Given a fucntion x^3 - a = 0, find x.

# Solution
a = 2
epsilon = 1e-4
x_0 = 1.0
x_1 = x_0 - x_0 - (x_0**3 - a) / (3 * x_0**2)

while abs(x_0 - x_1) > epsilon:
    x_0 = x_1
    x_1 = x_1 - x_1 - (x_1**3 - a) / (3 * x_1**2)

