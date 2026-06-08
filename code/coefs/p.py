from sympy import symbols, expand_trig, simplify, binomial, expand

# Define the variable
w = symbols('w')

def calculate_coefficients_of_gk(k):
    # Initialize g_k
    g_k = 0

    # Calculate g_k using the given expression
    for l in range(k + 1):
        g_k += binomial(2 * k + 1, k - l) * (1 - w)**(k - l) * (1 + w)**l

    # Multiply by the constant factor and simplify the expression
    g_k *= (1 + w)**(k + 1) / (2**(2 * k + 1))

    # Expand the expression to collect coefficients of cos(w)^l
    expanded_gk = expand(g_k)

    return expanded_gk

# Example: Calculate coefficients for k = 3
 # take k_value from sys.argv



import sys
k_value = int(sys.argv[1])
print(f"Calculating coefficients for k = {k_value}...")
result = calculate_coefficients_of_gk(k_value)
print(f"Coefficients of g_{k_value} as a sum of powers of cos(w):\n{result}")


def coefficients_cos_to_euler(coefficients_cos):
    # Using Euler's formula to express cos(w) in terms of e^(iwl)
    coefficients_euler = simplify(expand_trig(coefficients_cos.subs({w: symbols('l')})))

    return coefficients_euler

coefficients_euler = coefficients_cos_to_euler(result)
print(f"Coefficients in terms of e^(iwl): {coefficients_euler}")



# Define the variable
w = symbols('w l')

def coefficients_cos_to_euler(coefficients_cos, degree):
    # Using Euler's formula to express cos(w) in terms of e^(iwl)
    coefficients_euler = simplify(coefficients_cos.subs({w: exp(l * symbols('I'))}).expand().subs({exp(l * symbols('I')): cos(l * symbols('w'))}))

    return coefficients_euler

# Example coefficients in terms of cos(w)
coefficients_cos = -w**3/4 + 3*w/4 + 1/2

# Transform coefficients to be in terms of e^(iwl)
coefficients_euler = coefficients_cos_to_euler(coefficients_cos, 3)  # 3 is the degree of the polynomial in cos(w)
print(f"Coefficients in terms of e^(iwl): {coefficients_euler}")