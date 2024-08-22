from collections import defaultdict
import numpy as np
import subprocess
from scipy.optimize import minimize


def generate_non_singular_matrix(N):
    while True:
        matrix = np.random.randint(-6, 7, (N, N))
        if np.linalg.det(matrix) != 0:
            return matrix


def transpose_and_multiply(matrix):
    transpose = matrix.T
    product = np.dot(matrix, transpose)
    return transpose, product


def scale_and_round(matrix):
    scaled_matrix = np.round(matrix / 10).astype(int)
    return scaled_matrix


def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)


def generate_positive_definite_matrix(N):
    while True:
        matrix = generate_non_singular_matrix(N)
        transpose, product = transpose_and_multiply(matrix)
        final_matrix = scale_and_round(product)
        if is_positive_definite(final_matrix):
            return final_matrix


def f(x, W, a):
    return x.T @ W @ x + (x - a).T @ (x - a)


def gradient(f, x, W, a):
    grad = np.zeros_like(x)
    h = 1e-5
    for i in range(len(x)):
        x_h1 = np.copy(x)
        x_h2 = np.copy(x)
        x_h1[i] += h
        x_h2[i] -= h
        grad[i] = (f(x_h1, W, a) - f(x_h2, W, a)) / (2 * h)
    return grad


def gradient_descent(f, x0, W, a, epsilon, beta, lambda_):
    x = np.copy(x0)
    num_iterations = 0
    while True:
        grad = gradient(f, x, W, a)
        if np.max(np.abs(grad)) < epsilon:
            break
        t = beta
        while f(x - t * grad, W, a) > f(x, W, a):
            t *= lambda_
        x = x - t * grad
        num_iterations += 1
    return x, f(x, W, a), num_iterations


def create_latex_table(W, a):
    N = W.shape[0]
    latex = "\\begin{align*}\n"
    latex += f"f({', '.join(f'x_{{{j + 1}}}' for j in range(N))}) &= "
    linear_terms = [0] * N
    free_term = 0
    first_term = True
    line_length = 0

    def add_term(term):
        nonlocal latex, line_length, first_term
        if line_length + len(term) > 80:
            latex += " \\\\ \n& "
            line_length = 0
        if not first_term:
            latex += f" + {term}" if term[0] != '-' else f" {term}"
        else:
            latex += term
            first_term = False
        line_length += len(term)

    for i in range(N):
        for j in range(i, N):
            coef = W[i, j]
            if i == j:
                term = f"{coef + 1}x_{{{i + 1}}}^2"
            else:
                term = f"{2 * coef}x_{{{i + 1}}}x_{{{j + 1}}}"

            if coef != 0:
                add_term(term)

    for i in range(N):
        linear_terms[i] -= 2 * a[i]
        free_term += a[i] ** 2

    for i in range(N):
        if linear_terms[i] != 0:
            term = f"{linear_terms[i]}x_{{{i + 1}}}"
            add_term(term)

    if free_term != 0:
        add_term(f"{free_term}")

    latex += "\n\\end{align*}\n"
    return latex


def create_latex_document(variants, solutions, N, epsilon, beta, lambda_, with_solution=False):
    latex = "\\documentclass{article}\n"
    latex += "\\usepackage[utf8]{inputenc}\n"
    latex += "\\usepackage[russian]{babel}\n"
    latex += "\\usepackage{amsmath}\n"
    latex += "\\usepackage{geometry}\n"
    latex += "\\geometry{a4paper, margin=1in}\n"
    latex += "\\begin{document}\n"

    variant_pairs = list(zip(variants[0], variants[1]))

    for i, ((W, a), (minimum_x, minimum_value, final_x, final_value, num_iterations, x0)) in enumerate(
            zip(variant_pairs, solutions)):
        latex += f"\\section*{{\\textbf{{Вариант № {i + 1}}}}}\n"
        latex += "{Найти минимум функции:}\n"
        latex += create_latex_table(W, a)
        latex += "методом градиентного спуска с дроблением шага.\n"
        latex += "\\newline\n"
        latex += "В качестве начального приближения взять:\n"
        latex += "$$\n"
        latex += "\\begin{bmatrix}\n"
        latex += "\\\\\n".join(f"x_{{{j + 1}}}" for j in range(N)) + "\n"
        latex += "\\end{bmatrix}\n"
        latex += " = \n"
        latex += "\\begin{bmatrix}\n"
        latex += "\\\\\n".join(map(str, x0)) + "\n"
        latex += "\\end{bmatrix}\n"
        latex += "$$\n"
        latex += f"В качестве точности вычисления взять $\\varepsilon = {epsilon}.$\n"
        latex += "\\newline\n"
        latex += "В качестве критерия остановки:\n"
        latex += "$$\n"
        latex += "\\max\\limits_{1 \\leq j \\leq n} \\left| \\frac{\\partial f_0 (x^k)}{\\partial x_j} \\right| < \\varepsilon.\n"
        latex += "$$\n"
        latex += f"В качестве начальной длины шага взять $\\beta = {beta}.$\n"
        latex += "\\newline\n"
        latex += f"В качестве коэффициента дробления $\\lambda = {lambda_}.$\n"
        latex += "\\newline\n"

        if with_solution:
            latex += "\\textbf{Решение:}\n"
            latex += "\\newline\n"
            latex += "Минимум найден встроенной функцией в точке:\n"
            latex += "$$\n"
            latex += "\\begin{bmatrix}\n"
            latex += "\\\\\n".join(map(str, minimum_x)) + "\n"
            latex += "\\end{bmatrix}\n"
            latex += "$$\n"
            latex += f"Значение функции в точке:\n {minimum_value}.\n"
            latex += "\\newline\n"
            latex += "Минимум найденный методом градиентного спуска в точке:\n"
            latex += "$$\n"
            latex += "\\begin{bmatrix}\n"
            latex += "\\\\\n".join(map(str, final_x)) + "\n"
            latex += "\\end{bmatrix}\n"
            latex += "$$\n"
            latex += f"Значение функции в точке:\n {final_value}.\n"
            latex += "\\newline\n"
            latex += f"Число итераций:\n {num_iterations}.\n"

        if i < len(variants[0]) - 1:
            latex += "\\newpage\n"

    latex += "\\end{document}\n"
    return latex


def save_latex_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


def compile_latex_to_pdf(latex_filename):
    result = subprocess.run(["pdflatex", latex_filename], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error during LaTeX compilation:")
        print(result.stderr)
        with open("error.log", 'w') as f:
            f.write(result.stderr)
    else:
        print("PDF file created successfully.")

def generate_a(N):
    return np.random.randint(-10, 11, N)

def main():
    N = 6  # задано
    num_variants = 30  # задано
    epsilon = 0.01  # задано
    beta = 0.5  # задано
    lambda_ = 0.5  # задано

    variants = [[generate_positive_definite_matrix(N) for _ in range(num_variants)],
                [generate_a(N) for _ in range(num_variants)]]
    solutions = []

    for W, a in zip(variants[0], variants[1]):
        print(f"Generated matrix W:\n{W}\n")
        print(f"Generated a :\n{a}\n")
        x0 = np.round(np.random.rand(N), 3)
        print(f"Initial point x0: {x0}\n")

        result = minimize(f, x0, args=(W, a))
        minimum_x = result.x
        minimum_value = result.fun

        print(f"Minimum found by built-in function at point: {minimum_x}")
        print(f"Value of the function at this point: {minimum_value}\n")

        start_point = minimum_x + np.random.normal(0, 0.1, size=minimum_x.shape)
        final_x, final_value, num_iterations = gradient_descent(f, start_point, W, a, epsilon, beta, lambda_)

        print(f"Minimum found by gradient descent at point: {final_x}")
        print(f"Value of the function at this point: {final_value}")
        print(f"Number of iterations: {num_iterations}\n")

        solutions.append((minimum_x, minimum_value, final_x, final_value, num_iterations, x0))

    latex_content_without_solution = create_latex_document(variants, solutions, N, epsilon, beta, lambda_, with_solution=False)
    save_latex_file("tasks.tex", latex_content_without_solution)
    compile_latex_to_pdf("tasks.tex")

    latex_content_with_solution = create_latex_document(variants, solutions, N, epsilon, beta, lambda_, with_solution=True)
    save_latex_file("solutions.tex", latex_content_with_solution)
    compile_latex_to_pdf("solutions.tex")


if __name__ == "__main__":
    main()
