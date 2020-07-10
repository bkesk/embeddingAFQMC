class IntegralGenerator:
    def __init__(self,*args,**kwargs):
        pass

    def row(self,index,*args,**kwargs):
        return self.get_row(index,*args,**kwargs)

    def diagonal(self,*args,**kwargs):
        return self.get_diagonal(*args,**kwargs)
