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



def f(x, W):
    return x.T @ W @ x



def gradient(f, x, W):
    grad = np.zeros_like(x)
    h = 1e-5
    for i in range(len(x)):
        x_h1 = np.copy(x)
        x_h2 = np.copy(x)
        x_h1[i] += h
        x_h2[i] -= h
        grad[i] = (f(x_h1, W) - f(x_h2, W)) / (2 * h)
    return grad



def gradient_descent(f, x0, W, epsilon, beta, lambda_):
    x = np.copy(x0)
    num_iterations = 0
    while True:
        grad = gradient(f, x, W)
        if np.max(np.abs(grad)) < epsilon:
            break
        t = beta
        while f(x - t * grad, W) > f(x, W):
            t *= lambda_
        x = x - t * grad
        num_iterations += 1
    return x, f(x, W), num_iterations



def create_latex_table(W):

    N = W.shape[0]
    latex = "\\begin{equation}\n"
    latex += "f_0(x_1, x_2, \\ldots, x_n) = "

    terms = []
    for i in range(N):
        for j in range(i, N):
            coef = W[i, j]
            if coef != 0:
                if i == j:
                    if coef == 1:
                        term = f"x_{i + 1}^2"
                    elif coef == -1:
                        term = f"-x_{i + 1}^2"
                    else:
                        term = f"{coef} x_{i + 1}^2"
                else:
                    if coef == 1:
                        term = f"x_{i + 1} x_{j + 1}"
                    elif coef == -1:
                        term = f"-x_{i + 1} x_{j + 1}"
                    else:
                        term = f"{coef} x_{i + 1} x_{j + 1}"
                terms.append(term)


    latex += " + ".join(terms).replace(' + -', ' - ')
    latex += "\n\\end{equation}\n"

    return latex



def create_latex_document(variants, solutions, N, epsilon, beta, lambda_, with_solution=False):
    latex = "\\documentclass{article}\n"
    latex += "\\usepackage[utf8]{inputenc}\n"
    latex += "\\usepackage[russian]{babel}\n"
    latex += "\\usepackage{amsmath}\n"
    latex += "\\usepackage{geometry}\n"
    latex += "\\geometry{a4paper, margin=1in}\n"
    latex += "\\begin{document}\n"
    for i, (W, (minimum_x, minimum_value, final_x, final_value, num_iterations, x0)) in enumerate(zip(variants, solutions)):
        latex += f"\\section*{{\\textbf{{Вариант № {i + 1}}}}}\n"
        latex += "{Найти минимум функции:}\n"
        latex += create_latex_table(W)
        latex += "методом градиентного спуска с дроблением шага.\n"
        latex += "\\newline\n"
        latex += "В качестве начального приближения взять:\n"
        latex += f"$$\n"
        latex += f"({', '.join(f'x_{{{j + 1}}}' for j in range(N))}) = ({', '.join(map(str, x0))})\n"
        latex += f"$$\n"
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
            latex += f"$$\n"
            latex += f"({', '.join(map(str, minimum_x))})"
            latex += f"$$\n"
            latex += f"Значение функции в точке:\n {minimum_value}.\n"
            latex += "\\newline\n"
            latex += "Минимум найденный методом градиентного спуска в точке:\n"
            latex += f"$$\n"
            latex += f"({', '.join(map(str, final_x))})"
            latex += f"$$\n"
            latex += f"Значение функции в точке:\n {final_value}.\n"
            latex += "\\newline\n"
            latex += f"Число итераций:\n {num_iterations}.\n"
        if i < len(variants) - 1:
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



def main():
    N = 3  #задается
    num_variants = 3  #задается
    epsilon = 0.01 #задается
    beta = 0.5 #задается
    lambda_ = 0.5 #задается

    variants = [generate_positive_definite_matrix(N) for _ in range(num_variants)]
    solutions = []

    for W in variants:
        print(f"Generated matrix W:\n{W}\n")
        x0 = np.round(np.random.rand(N),3)
        print(f"Initial point x0: {x0}\n")

        result = minimize(f, x0, args=(W,))
        minimum_x = result.x
        minimum_value = result.fun

        print(f"Minimum found by built-in function at point: {minimum_x}")
        print(f"Value of the function at this point: {minimum_value}\n")

        start_point = minimum_x + np.random.normal(0, 0.1, size=minimum_x.shape)
        final_x, final_value, num_iterations = gradient_descent(f, start_point, W, epsilon, beta, lambda_)

        print(f"Minimum found by gradient descent at point: {final_x}")
        print(f"Value of the function at this point: {final_value}")
        print(f"Number of iterations: {num_iterations}\n")

        solutions.append((minimum_x, minimum_value, final_x, final_value, num_iterations, x0))

    latex_content_without_solution = create_latex_document(variants, solutions, N, epsilon, beta, lambda_,
                                                           with_solution=False)
    save_latex_file("tasks_only.tex", latex_content_without_solution)
    compile_latex_to_pdf("tasks_only.tex")

    latex_content_with_solution = create_latex_document(variants, solutions, N, epsilon, beta, lambda_,
                                                        with_solution=True)
    save_latex_file("tasks_with_solutions.tex", latex_content_with_solution)
    compile_latex_to_pdf("tasks_with_solutions.tex")


if __name__ == "__main__":
    main()
