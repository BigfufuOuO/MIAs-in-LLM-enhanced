def convert_input_to_matrix():
    matrix = []
    while True:
        try:
            line = input().strip()
            if not line:
                break
            row = [float(num) for num in line.split()]
            matrix.append(row)
        except EOFError:
            break
    if len(matrix) == 1:
        return matrix[0]
    return matrix


result = convert_input_to_matrix()
print(result)