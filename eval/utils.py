from .evaluator_clf import BinClf_Evaluator
from .evaluator_clf import MultiClf_Evaluator


def load_evaluator(task, *args, **kws):
    if task == 'clf':    
        if kws['binary_clf']:
            evaluator = BinClf_Evaluator(**kws)
        else:
            evaluator = MultiClf_Evaluator(**kws)
    else:
        pass
    
    return evaluator
