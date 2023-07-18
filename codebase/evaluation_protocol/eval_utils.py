class EvalWrapper:
    def __init__(self, model):
        self.model = model
    
    def new_eval(self, *args, **kwargs):
        pass

    def get_action(self, *args, **kwargs):
        return self.model.get_action(*args, **kwargs)
    
    def update_history(self, *args, **kwargs):
        pass
