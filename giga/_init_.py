from importlib import import_module

def get(args):
    model_name = args.model_name + '_Model'
    module_name = 'model.' + model_name.lower()
    module = import_module(module_name)

    return getattr(module, model_name)