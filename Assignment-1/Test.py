import numpy as np
class Lin:
    def ok(self):
        print('ok parent')
    

class Grad:
    def grad(self):
        print('Gradient parent')
    
    def lmao(self):
        print('lmao parent')
        
        
class Grad_(Grad):
    def lmao(self):
        super().grad()
        
    def grad(self):
        print('Gradient child')


obj = Grad_()
obj.lmao()

matrix = np.array( [[1,2,3,7],
                    [8,4,6,3],
                    [6,2,4,9]])
model_parameter = np.array([ 3,
                            -1,
                            -2,
                            4])

new_parameter = np.zeros(len(matrix))
            
for j in range(len(model_parameter)):
    jth_parameter = model_parameter[j]
    model_parameter[j] = 0
    
    print(np.dot(matrix, model_parameter))

    model_parameter[j] = jth_parameter
    

new_parameter