import statistics as stat

"""
scaling = Xi-Xmin/Xmax-Xmin
"""

def matrix_transpose(matrixA: list[list]):
        transposed_matrix = []
        for i in range(len(matrixA[0])):
            transposed_matrix.append([])
        for x in range(len(matrixA)):
            for y in range(len(matrixA[x])):
                transposed_matrix[y].append(matrixA[x][y])
        return transposed_matrix

class knn:
    def __init__(self, factors:list[list], outputs:list, k:int):
        self.factors = factors
        self.outputs = outputs
        self.k = k
    

    def predict(self, data: list):
        points = [list(x) for x in self.factors]
        scaled_data = list(data)
        for i in range(len(points)):
             temp_max = max(points[i])
             temp_min = min(points[i])
             for j in range(len(points[i])):
                  if temp_max-temp_min == 0:
                       points[i][j] = 0
                       continue
                  points[i][j] = (points[i][j]-temp_min)/(temp_max-temp_min)
             if temp_max-temp_min == 0:
                  scaled_data[i] = 0
                  continue
             scaled_data[i] = (scaled_data[i]-temp_min)/(temp_max-temp_min)
             
        points = matrix_transpose(points)
        
        distances = []
        
        for i in range(len(points)):
            distance = 0
            for j in range(len(points[i])):
                 distance += (scaled_data[j]-points[i][j])**(2)
            distances.append(distance**(1/2))
        
        points_and_class = []
        
        for i in range(len(distances)):
             points_and_class.append(([distances[i]], self.outputs[i]))
        
        results = []

        points_and_class = sorted(points_and_class)
        
        for i in range(len(points_and_class[:self.k])):
             results.append(points_and_class[i][1])
        
        return stat.multimode(results)
