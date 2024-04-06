class Test:
    def __init__(self):
        self._vars = 1
    
    @property
    def vars(self):
        return self._vars

    @vars.setter
    def vars(self, value):
        self._vars = value
        print(1)


t = Test()
t.vars = 2