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