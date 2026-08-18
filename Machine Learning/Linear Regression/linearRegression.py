# One factor one output model
class simpleLinearRegression:
    def __init__(self, inputs: list, outputs: list):
        self.inputs = inputs
        self.outputs = outputs
        self.slope = 0
        self.intercept = 0


    def calibrate(self):
        input_mean = sum(self.inputs)/len(self.inputs)
        output_mean = sum(self.outputs)/len(self.outputs)    

        inputs_sub_mean = [inputVal - input_mean for inputVal in self.inputs]
        outputs_sub_mean = [outputVal - output_mean for outputVal in self.outputs]

        input_sub_mean_squared = [IsubM ** 2 for IsubM in inputs_sub_mean] 

        input_output_sub_mean_multiplied = []
        for i in range(len(inputs_sub_mean)):
            input_output_sub_mean_multiplied.append(inputs_sub_mean[i]*outputs_sub_mean[i])

        self.slope = sum(input_output_sub_mean_multiplied)/sum(input_sub_mean_squared)
        self.intercept = output_mean - (input_mean*self.slope)

    def get_slope(self):
        return self.slope
    
    def get_intercept(self):
        return self.intercept

    def run_prediction(self, x):
        if self.slope == 0 and self.intercept == 0:
           self.calibrate()
        return (self.slope*x)+self.intercept
    
    def add_data(self, point:tuple):
        self.inputs.append(point[0])
        self.outputs.append(point[1])
        self.calibrate()


def matrix_multiply(matrixA: list[list], matrixB: list[list]):
    firstRowLengthA = len(matrixA[0])
    for row in matrixA:
        if len(row) != firstRowLengthA:
            print('Invalid Matrix A')
            return 
    firstRowLengthB = len(matrixB[0])
    for row in matrixB:
        if len(row) != firstRowLengthB:
            print('Invalid Matrix B')
            return 
    if firstRowLengthA != len(matrixB):
        print('Cannot multiply given matrices')
        return
    result_matrix = []
    rows = matrixA
    cols = []
    for i in range(len(matrixB[0])):
        cols.append([])
    for col in matrixB:
        for i in range(len(col)):
            cols[i].append(col[i])
    for i in range(len(rows)):
        new_row = []
        for j in range(len(cols)):
            new_row.append(0)
        result_matrix.append(new_row)
    for i in range(len(result_matrix)):
        for j in range(len(result_matrix[i])):
            current_row = rows[i]
            current_col = cols[j]
            row_items_times_cols_items = []
            for o in range(len(current_row)):
                row_items_times_cols_items.append(current_row[o]*current_col[o])
            result_matrix[i][j] = sum(row_items_times_cols_items)
    return result_matrix


def matrix_transpose(matrixA: list[list]):
    transposed_matrix = []
    for i in range(len(matrixA[0])):
        transposed_matrix.append([])
    for x in range(len(matrixA)):
        for y in range(len(matrixA[x])):
            transposed_matrix[y].append(matrixA[x][y])
    return transposed_matrix


def matrix_inverse(matrix: list[list]): #ChatGPT generated, my math isn't good enought yet*
    n = len(matrix)

    # Check square matrix
    for row in matrix:
        if len(row) != n:
            print("Matrix must be square")
            return None

    # Create augmented matrix [A | I]
    augmented = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix[i][j])
        for j in range(n):
            row.append(1 if i == j else 0)
        augmented.append(row)

    # Perform Gauss-Jordan elimination
    for i in range(n):
        pivot = augmented[i][i]

        if pivot == 0:
            # Try to swap with a lower row
            for r in range(i + 1, n):
                if augmented[r][i] != 0:
                    augmented[i], augmented[r] = augmented[r], augmented[i]
                    pivot = augmented[i][i]
                    break
            else:
                print("Matrix is singular and cannot be inverted")
                return None

        # Normalize pivot row
        for j in range(2 * n):
            augmented[i][j] /= pivot

        # Eliminate column values
        for r in range(n):
            if r != i:
                factor = augmented[r][i]
                for c in range(2 * n):
                    augmented[r][c] -= factor * augmented[i][c]

    # Extract inverse matrix
    inverse = []
    for i in range(n):
        inverse.append(augmented[i][n:])

    return inverse



# Multiple factors one output model
class multipleLinearRegression:
    def __init__(self, factors: list[list], outputs: list):
        self.factors = factors
        self.output = outputs
        # Weights: β= ( (x^t)(x) )^-1 * (x^t)(y)
        X = matrix_transpose(self.factors)
        XT = matrix_transpose(X)
        XTX = matrix_multiply(XT, X)
        XTX_Inverse = matrix_inverse(XTX)
        y = [[val] for val in self.output]
        XTy = matrix_multiply(XT, y)
        betas = matrix_multiply(XTX_Inverse, XTy)
        self.weights = [b[0] for b in betas]


        self.means = [sum(factor)/len(factor) for factor in factors]
        betas_means = []
        for i in range(len(self.weights)):
            betas_means.append(self.weights[i]*self.means[i])
        self.intercept = (sum(outputs)/len(outputs))-sum(betas_means)

    def predict(self, factors):
        result = 0
        for i in range(len(self.weights)):
            result += self.weights[i]*factors[i]
        return result+self.intercept
            
if __name__ == '__main__':
    pass