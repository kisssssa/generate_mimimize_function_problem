

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
            return matrix, transpose, product, final_matrix


N = 3
matrix, transpose, product, final_matrix = generate_positive_definite_matrix(N)
print("Конечная матрица после деления на 10 и округления:")
print(final_matrix)
